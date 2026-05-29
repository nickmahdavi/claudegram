# BYOK — implementation notes & lessons (parked)

This effort (per-user Claude credentials for the bot) was cut in favor of a more convenient
solution. This file distills the design, the hard-won gotchas, and the OAuth findings so a new
implementation doesn't have to relearn them. **None of this is on `master`.**

## Where the actual code is
- **Full implementation:** committed on the **`byok` branch** at `675a293`
  ("bring your own credentials (key/subscription/pool)"), 10 source files, ~1,400 lines.
  - Retrieve a file without checking out: `git show byok:src/claudegram/credentials.py`
  - Whole diff vs its base: `git diff 95f2513 byok`
- **Tests:** live untracked on disk under `tests/` (the `.gitignore` `/*/` rule ignores the dir).
  `test_secrets.py`, `test_credentials.py`, `test_bot_auth_isolation.py`, `test_oauth.py`.
  44 passing at peak. Not committed anywhere.
- It went through **four review rounds** (security, concurrency, PTB-correctness, tests, plus two
  open-ended Opus passes). The findings below are the residue of those.

---

## What it did

Replace the single shared Anthropic key with **per-request credential resolution**. Three kinds:
`USER_API_KEY` (user's own key), `OAUTH_SUBSCRIPTION` (Claude Pro/Max), `POOL` (uses the bot's
own key). Per-chat **billing mode** decides who pays in groups.

## Architecture (reusable regardless of transport)

The whole feature hangs off **one integration point**: `model.complete()` already takes a
`client` arg, and `on_ping` was the only caller passing the single shared `self.client`. So:

1. `on_ping` calls `CredentialStore.resolve_credential(user_id, chat_id, is_private)` → a
   `Credential | None`.
2. `None` → polite refusal (and crucially, **do not** trip the failure streak).
3. Otherwise `await CredentialStore.client_for(cred)` → a configured `AsyncClient`, passed to
   `complete()`.

`CredentialStore` (in `credentials.py`) owns:
- **Persistence** (mirrors the existing `store.py` JSON conventions: atomic tmp+`os.replace`,
  per-row try/except, lazy file creation):
  - `credentials.json` — `user_id -> {kind, enc_secret, …}`, **mode 0o600**, secrets stored
    **only** as Fernet ciphertext.
  - `pool_users.json` — sorted list of user ids.
  - `chat_billing.json` — `chat_id -> {mode, designated_user_id?}`.
- **Resolution precedence** (per user): own API key > own OAuth > pool > nothing.
  - DM: resolve for the messaging user.
  - Group: `TRIGGERING_USER` (default — the pinger pays) or `CHAT_DESIGNATED` (one user's
    credential covers the chat), admin-set via `/billing`.
- **Client cache** keyed by `(user_id, kind, secret-fingerprint)` so httpx pools are reused;
  rebuild on credential change. (See the eviction gotcha below.)

**Encryption** (`secrets.py`): thin Fernet wrapper. Master key from `CREDENTIAL_ENC_KEY`. If it's
missing/malformed, **disable credential storage, don't crash** (`Secrets.available == False`, the
commands refuse, the pool still works). A rotated key → stored rows fail to decrypt → dropped on
load (users re-enter keys). Never log plaintext.

**Wiring** (`__main__.py`): build `Secrets(config.credential_enc_key)` and
`CredentialStore(data_dir, secrets, config)`, pass into `Bot`. The bot keeps a pool client built
from `claude_api_key` (still required).

## Commands
- `/setkey <key>` — DM-only; delete the inbound message immediately; validate with a cheap
  1-token call; store encrypted. `/forgetkey`, `/keystatus`.
- `/allow <user_id|reply>`, `/disallow`, `/poollist` — admin pool management.
- `/billing triggering|designated [user]` — admin, group; per-chat payer mode.
- `/login`, `/code` — OAuth (the dead end; see below).

---

## Hard-won gotchas (the valuable part — these bit us)

1. **Handler-group secret leak.** `on_message` and `on_ping` are *separate* PTB handler groups;
   `return`-ing from one does **not** stop the other (only `ApplicationHandlerStop` does). Since
   every DM is also a ping, a bare key pasted in a DM was processed by `on_ping` even after
   `on_message` dropped it — and the bot's reply persisted the key via its quoted `Reply`.
   **Fix:** the secret guard must raise `ApplicationHandlerStop` (and/or live in both handlers).

2. **Per-user failure isolation (highest-value bug).** A user's *own* credential failing — AUTH,
   **out-of-credit (402)**, rate-limit, etc. — must **not** feed the chat's failure-streak /
   admin-alert machinery (that tracks only the bot's pool key) and must **not** show a "bot is
   down" message. Attribute it to the user instead. Out-of-credit is the single most common BYO
   failure and is the easiest to misattribute. Symmetrically, only a **pool** success should clear
   the streak — otherwise mixed-credential groups flap recovery/outage alerts.

3. **Markdown injection via display names.** Telegram `full_name` is user-controlled; interpolated
   into `parse_mode="Markdown"` an unbalanced `_ * ` [` makes the *whole send* 400. Escape with
   `telegram.helpers.escape_markdown(name, version=1)` (placed outside backticks) or drop
   `parse_mode`. The existing `_notify_admins_*` code already avoids `parse_mode` for this reason —
   follow that convention.

4. **Don't close evicted clients eagerly.** Closing an evicted `AsyncClient` can abort an
   in-flight `messages.create` on another coroutine, and a fire-and-forget `create_task(close)`
   can be GC'd before it runs. Just drop the reference (GC reclaims it); close everything at
   shutdown via a `post_shutdown` hook (`CredentialStore.aclose()`).

5. **Markdown-breaking URLs.** OAuth authorize URLs are full of `_` (random PKCE/state); under
   legacy Markdown they 400 ~half the time. Send URLs with **no** `parse_mode` (Telegram
   auto-links them and the content can't corrupt the message).

6. **`message.delete()` semantics.** Bots *can* delete a user's incoming message in a **private**
   chat (verified, PTB 22.7); in **groups** only with `can_delete_messages`. Gate any "I deleted
   that" wording on the actual return value, else you lie when the delete failed.

7. **Secrets at rest.** Write `credentials.json` with `0o600` *and* `os.fchmod` (the `os.open`
   mode arg only applies on creation, so a stale looser tmp file would slip through). Only
   ciphertext on disk; mask secrets in `/keystatus` and logs (last-4 only).

8. **`_looks_like_secret` guard.** Match the command as a *first-token* exact check (so `/coder`
   isn't caught) and scan for an `sk-ant-` token *anywhere* in the message (not just a prefix).

---

## The OAuth finding (most important input to the new design)

Subscription (Pro/Max) OAuth was implemented end-to-end and **tested live**. Conclusion: it's a
dead end via the raw Messages API.

- The OAuth token **authenticates** — `/v1/messages` resolves it to an org (the response carries
  `anthropic-organization-id`). It is **not** a 401.
- But **every completion returns 429 with no `anthropic-ratelimit-*` headers**. A real quota 429
  carries those headers; their absence means it's a **policy block**, not throttling. (Confirmed
  in logs: the API-key validation call got `200` on org `56666bd0…` with full quota headers; the
  OAuth completion got `429` on org `197489dc…` with none.)
- **Mechanism:** the subscription path is only honored for requests that look like **Claude Code**
  (its specific identity system prompt). The bot sends its own chat system prompt → blocked.
- **Implication:** making subscription auth work against the raw API means **impersonating Claude
  Code's system prompt** = a ToS gray area we chose not to cross.
- **The sanctioned route is the Claude Agent SDK** (`claude-agent-sdk`) — the library Claude Code
  itself is built on, which carries the right identity/headers for subscription auth. If the new
  implementation wants subscription support, route OAUTH-kind completions through the Agent SDK,
  **not** raw `messages.create`. Caveat: the Agent SDK is an agent loop, not a one-shot call, so
  the request/response shape and token-accounting (`ClaudeResponse`) won't map 1:1 — it's a second
  completion path, not a drop-in client swap.
- For reference, the reverse-engineered (unverified) OAuth constants we used:
  - client_id `9d1c250a-e61b-44d9-88ed-5944d1962f5e`
  - authorize `https://claude.ai/oauth/authorize`, token `https://console.anthropic.com/v1/oauth/token`
  - redirect `https://console.anthropic.com/oauth/code/callback`
  - scope `org:create_api_key user:profile user:inference`, beta header `oauth-2025-04-20`, PKCE S256

**Bottom line for the new approach:** BYO **API key** + **admin pool** + per-chat **billing** all
work and are well-tested — lift those. Treat **subscription auth** as a separate, Agent-SDK-shaped
problem (or drop it).

---

## File map (on `byok` branch)
| File | Role |
|---|---|
| `secrets.py` | Fernet encryption wrapper; disable-not-crash on missing key. |
| `credentials.py` | `Credential` / `CredentialKind` / `BillingMode`; `CredentialStore` (persistence, resolution, client cache, validate). |
| `oauth.py` | PKCE login/exchange/refresh. **The part to drop or replace with the Agent SDK.** |
| `error.py` | Credential-related user-facing messages (no-cred refusal, per-user auth/credit attribution). |
| `config.py` | `credential_enc_key`, plus OAuth fields. |
| `bot.py` | `on_ping` resolution, `_handle_completion_error` isolation, `on_message` secret guard, all the commands. |
| `__main__.py` | Builds `Secrets` + `CredentialStore`, passes into `Bot`. |

## Highest-value tests to re-create (if reusing)
- AUTH/credit isolation: user-owned failure does NOT trip the streak; pool failure does.
- Resolution precedence table (DM/group × billing mode × has key/pool/nothing).
- Secrets round-trip + rotation (undecryptable rows dropped).
- `on_message` secret-drop (delete + `ApplicationHandlerStop` + nothing persisted).
- Client factory headers (`X-Api-Key` for key; `Authorization: Bearer` + beta header for OAuth).
