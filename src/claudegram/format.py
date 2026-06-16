"""Markdown -> Telegram HTML conversion for outgoing bot replies.

Claude emits CommonMark (`**bold**`, `` `code` ``, bullet lists, fenced code).
Telegram doesn't parse it natively, so without a parse_mode the markup shows up
literally in the chat. Telegram offers two parse modes:

  - MarkdownV2: a different markdown dialect, with a long list of characters
    that must be backslash-escaped in plain text. Easy to get wrong.
  - HTML: forgiving about nesting/whitespace; only `<`, `>`, `&` need escaping
    in plain text. Tag set is a small subset (b/i/s/u/code/pre/a/blockquote/
    tg-spoiler).

We go via HTML.

Limitations: `<pre>` has no syntax highlighting, headers promote to `<b>`,
list markers become Unicode bullets (Telegram has no native list structure),
horizontal rules and tables fall through as plain text."""

import re

# Sentinels for stashing code spans before we touch the rest. NUL is illegal in
# JSON-serialized strings and shouldn't appear in any sane reply; if it somehow
# does, the replacement keys still don't collide because they include a counter.
_CODE_SLOT = "\x00CODE{}\x00"
_INLINE_SLOT = "\x00ICODE{}\x00"


def _escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def md_to_html(text: str) -> str:
    """Convert CommonMark-ish markdown to Telegram HTML."""
    # 1. Stash fenced code blocks so their bodies skip all later rewriting.
    # The opening fence's language tag (` ```python `) is dropped.
    fences: list[str] = []
    def _stash_pre(m: re.Match) -> str:
        body = m.group(1).rstrip("\n")
        fences.append(f"<pre>{_escape_html(body)}</pre>")
        return _CODE_SLOT.format(len(fences) - 1)
    text = re.sub(r"```[^\n]*\n?(.*?)```", _stash_pre, text, flags=re.DOTALL)

    # 2. Stash inline code spans.
    inlines: list[str] = []
    def _stash_code(m: re.Match) -> str:
        inlines.append(f"<code>{_escape_html(m.group(1))}</code>")
        return _INLINE_SLOT.format(len(inlines) - 1)
    text = re.sub(r"`([^`\n]+)`", _stash_code, text)

    # 3. Escape HTML on whatever survives.
    text = _escape_html(text)

    # 4. Bold (** or __) before italic so the doubled markers consume first.
    text = re.sub(r"\*\*([^*\n]+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__([^_\n]+?)__", r"<b>\1</b>", text)

    # 5. Italic (* or _). Lookarounds keep us from grabbing list markers
    # (`* item`) or intra-word underscores (`foo_bar_baz`).
    text = re.sub(r"(?<![*\w])\*([^*\n]+?)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<![_\w])_([^_\n]+?)_(?!\w)", r"<i>\1</i>", text)

    # 6. Strikethrough.
    text = re.sub(r"~~([^~\n]+?)~~", r"<s>\1</s>", text)

    # 7. Inline links.
    text = re.sub(r"\[([^\]\n]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', text)

    # 8. Headers -> bold (Telegram has no header element).
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)

    # 9. Bullet markers -> Unicode bullet, preserving indent.
    text = re.sub(r"^([ \t]*)[*-][ \t]+", r"\1• ", text, flags=re.MULTILINE)

    # 10. Re-insert stashed code.
    for i, c in enumerate(inlines):
        text = text.replace(_INLINE_SLOT.format(i), c)
    for i, c in enumerate(fences):
        text = text.replace(_CODE_SLOT.format(i), c)

    return text


def chunk_markdown(text: str, limit: int) -> list[str]:
    """Split markdown into <=limit-char chunks at the latest sensible boundary
    (paragraph, then line, then space, then hard slice). Each chunk passes
    through md_to_html independently, so inline spans straddling a chunk
    boundary degrade to literal text in one chunk but never produce
    half-open HTML tags."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = rest.rfind("\n\n", 0, limit + 1)
        if cut <= 0:
            cut = rest.rfind("\n", 0, limit + 1)
        if cut <= 0:
            cut = rest.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit
        chunks.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        chunks.append(rest)
    return chunks
