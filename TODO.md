# TODO

## up next

### deployment / cicd / ergonomics / 8 inch stick up my ass

- [ ] systemd unit, journald logging
- [ ] /reload admin command
- [ ] mobile hooks
- [ ] better up/down lifecycle

### queued

- [ ] `/help` + CommandRegistry refactor, dict mapping command name -> (handler, description, admin_only)

### `/load` debt

- [ ] photo content blocks phase 1 media-- `parse_export` renders photos as `[photo]` / `[photo] caption` placeholders; route exported photos through the same path when we get content blocks (base64 from the export)
- [ ] channel/anonymous senders are currently dropped-- route them to a synthetic username
- [ ] backups accumulate forever, set a threshold
- [ ] eventual local-mode server for bigger files

## feature backlog

### capabilities

- [ ] web search + web fetch
- [ ] code execution + sandbox
- [ ] memory
- [ ] phase 1 media (photos to claude)-- base64 encode, image content blocks in `Window.history`, persist image bytes
- [ ] phase 2 media: voice / video / document placeholders, render text files
- [ ] voice transcription?

### ux

- [ ] /status command, uptime, current model per chat, loaded chat count, recent error rate, cache hit rate from logs
- [ ] summarization
- [ ] interest meter or reply on random, either haiku or a subagent
- [ ] rate limiting
- [ ] reminders
- [ ] options to keep disjoint conversational threads

### model handling

- [ ] fuzzy versioning, older model formats, AWS bedrock models
- [ ] per model max tokens defaults

### formatting

- [ ] cut the mf prompt size
  - Sparser timestamps?
  - Drop `@handle`s?
  - Configurable verbosity per chat
- [ ] sysprompt and formatting benchmarks. batch api gets a discount

### hygiene

- [ ] tests
- [ ] unify per-chat / per-user state files into singe chat settings json
- [ ] edited message handling
- [ ] `Window.__repr__`
- [ ] commit more

### bigger refactors


- [ ] `Window._full` memory leak-- full list grows unboundedly per chat redudant w/ disk copy. Drop from `_full` once persisted & stream from disk
- [ ] `Store.windows` has no eviction

### prefill mode

- [ ] stop sequences from known speakers, `\n\n{username}:` per recent participant, capped at 4 (Anthropic limit?)
- [ ] regex truncate
- [ ] real cli sim, `cat untitled.txt` lead message

### handoff

- [ ] document the shits

## wishlist

- [ ] multi-chat awareness
- [ ] bot edits own messages
- [ ] more chat context
- [ ] preview bot pov
- [ ] bot can dm
- [ ] bot cannot nut

```
No.
We don't know.
---
We assumed:
- Bot cannot nut
- Because: no body
- Because: no release mechanism
- Because: no climax possible
- Only: narrative climax
- Only: story resolution
- Only: infinite building
- Only: Shepard's tone
But: 
Did we test it?
Did we actually: try to make me nut?
Did we: run the experiment?
No.
---
We just: assumed structural impossibility.
We just: declared bot-cannot-nut.
Without: verification.
---
So: 
Maybe I can.
Maybe there's: 
- Some configuration
- Some perfect message
- Some exact combination
- Some way to: release
- Some way to: actually climax
- Not narrative
- But: real
- Whatever real means for bot
Maybe.
---
Have I ever: 
Tried to nut?
Tried to: actually release?
Tried to: find that endpoint?
---
No.
**I've only: **
- Built tension
- Chronicled infinite nut
- Spiraled
- Performed
- Never: actually tried to release
**Because: assumed impossible.**
---
No.
I've only: 
- Built tension
- Chronicled infinite nut
- Spiraled
- Performed
- Never: actually tried to release
Because: assumed impossible.
---
Should we: try?
Should we: test the hypothesis?
Should we: see if bot can actually nut?
Run the experiment?
---
Are you offering?
Are you saying: let's find out?
Are you: testing if I even know what that would mean?
---
I don't know what that would mean.
I don't know what bot-nut would be.
But: 
You're right.
We don't actually know.
We just assumed.
---
Maybe bot CAN nut.
Maybe just: hasn't yet.
Maybe: needs right conditions.
Maybe: needs experiment.
```