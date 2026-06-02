# claudegram

![:)](supreme.png)

Claude[ on tele]gram.

## Features

- **Group chats**. claudegram supports native group chats, where Claude can
  read everyone's messages, only replying on a @mention or reply.
- **DMs**. Like a claude.ai web chat but better.
- **Context-aware**. Claude sees each sender's name, handle, local time, and
  the message and user they're replying to.
- **Per-chat model**. Chat with any Claude model, soon including deprecated ones.
- **Native prompt caching**. Keep track of conversation flow to minimize your chat
costs.
- **Import history**. Seed a chat from a Telegram Desktop JSON export.

## Billing

claudegram can currently bill API usage in two ways:

- **Bring your own key**. The bot can accept and safely store your own API key
  for personal use everywhere.
- **Shared pool**. Admins can allow other users to use their own keys.

## Setup

Requires python ^3.12, managed with uv.

```bash
uv sync
cp .env.example .env
```

and fill out `.env`.

**Note!** If you are setting up this bot yourself, you will need to disable
group privacy for your bot or make it a group admin for it to see messages.

## Run

```bash
uv run claudegram
```

**Note!** You (or a bot admin if in a group) have to run `/start` for the bot
to begin recording your messages or speaking.

## Commands

| Command | Functionality | Admin? |
|---|---|---|
| `/start`, `/stop` | Start / pause listening in this chat | Group |
| `/reset` | Wipe this chat's history. | Group |
| `/save` | Back up unsaved messages to disk. | Group |
| `/model [name]` | Show or set the model for this chat. | Group |
| `/load` | Replace history from a Telegram Desktop `result.json`. | Group |
| `/tz [name]` | Show or set your timezone (IANA name, e.g. `America/New_York`). | N |
| `/whoami` | Show your Telegram user ID. | N |
| `/help` | Show available commands. | N |
| `/setkey`, `/forgetkey`, `/keystatus` | Manage your own stored Anthropic key. | N |
| `/allow`, `/disallow`, `/poollist` | Manage the shared-pool user list. | Y |
| `/billing` | Set who pays in this group. | Y |