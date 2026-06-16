"""Markdown -> Telegram HTML conversion for outgoing bot replies.

Claude emits CommonMark (`**bold**`, `` `code` ``, bullet lists, fenced code).
Telegram doesn't parse it natively, so without a parse_mode the markup shows up
literally in the chat. We go via HTML rather than Telegram's MarkdownV2 because
the escape rules are saner (only `<`, `>`, `&` need escaping in plain text) and
nesting is more forgiving.

Limitations / known asymmetries with CommonMark:
- `<pre>` has no syntax highlighting (Telegram doesn't support it).
- Headers promote to `<b>`. List markers become Unicode bullets (Telegram has
  no native list shape).
- Horizontal rules and tables fall through as plain text.
- Dunder-style identifiers like `__init__` get treated as bold per CommonMark
  spec; if that hurts in practice we can flip `__` bold off and require `**`.
- Deeply interleaved span markers (e.g. ``~~**x~~**``) can produce mis-nested
  HTML that Telegram rejects -- the converter validates the tag stack at the
  end and falls back to the original markdown text if it's malformed, so the
  message still goes through (as plain text) rather than crashing on send."""

import re
import secrets

_ALLOWED_SCHEMES = ("http://", "https://", "mailto:", "tel:")
# Tags we emit + their plain counterparts. Used by _is_balanced to verify the
# final HTML actually nests cleanly before we hand it to Telegram.
_TAG_RE = re.compile(
    r"<(/?)(b|i|s|u|code|pre|a|blockquote|tg-spoiler)\b[^>]*>",
    re.IGNORECASE,
)


def _escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _extract_url(text: str, start: int) -> int:
    """Find the end index (the matching `)`) of a markdown URL that begins at
    `start`. Permits balanced inner parens (so a Wikipedia disambiguation link
    like `https://en.wikipedia.org/wiki/Foo_(bar)` parses cleanly) but bails on
    any whitespace or newline before the close, since markdown URLs can't
    contain those. Returns -1 if no valid close is found."""
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            if depth == 0:
                return i
            depth -= 1
        elif c.isspace():
            return -1
        i += 1
    return -1


def _is_safe_url(url: str) -> bool:
    """Whitelist of schemes Telegram will accept and we want to ship. Anything
    else (javascript:, data:, file:, ...) falls back to literal text."""
    return any(url.startswith(s) for s in _ALLOWED_SCHEMES)


def _is_balanced(html: str) -> bool:
    """Walk the emitted tags as a stack; true iff every open has a matching
    close in LIFO order. Cheap sanity check before we hand off to Telegram --
    catches the mis-nested-span case (`<s><b>x</s></b>`) cleanly without
    needing a real HTML parser."""
    stack: list[str] = []
    for m in _TAG_RE.finditer(html):
        is_close, tag = m.group(1), m.group(2).lower()
        if is_close:
            if not stack or stack[-1] != tag:
                return False
            stack.pop()
        else:
            stack.append(tag)
    return not stack


def md_to_html(text: str) -> str:
    """Convert CommonMark-ish markdown to Telegram HTML.

    If the result fails the HTML balance check (e.g. interleaved markers
    produced mis-nested tags), returns the HTML-escaped original so `_send`
    ships it as plain rather than having Telegram reject it."""
    original = text
    # Per-conversion random sentinel prefix: prevents literal `\x00CODE0\x00`
    # in user input from colliding with our placeholder during str.replace.
    sentinel = f"\x00FMT_{secrets.token_hex(8)}_"
    def code_slot(n: int) -> str:    return f"{sentinel}CODE_{n}\x00"
    def inline_slot(n: int) -> str:  return f"{sentinel}ICODE_{n}\x00"
    def link_slot(n: int) -> str:    return f"{sentinel}LINK_{n}\x00"

    # 1. Stash fenced code first; bodies skip everything else. The language
    # tag on the opening fence (` ```python `) is dropped -- Telegram <pre>
    # doesn't render it anyway.
    fences: list[str] = []
    def _stash_pre(m: re.Match) -> str:
        body = m.group(1).rstrip("\n")
        fences.append(f"<pre>{_escape_html(body)}</pre>")
        return code_slot(len(fences) - 1)
    text = re.sub(r"```[^\n]*\n?(.*?)```", _stash_pre, text, flags=re.DOTALL)

    # 2. Stash inline code.
    inlines: list[str] = []
    def _stash_code(m: re.Match) -> str:
        inlines.append(f"<code>{_escape_html(m.group(1))}</code>")
        return inline_slot(len(inlines) - 1)
    text = re.sub(r"`([^`\n]+)`", _stash_code, text)

    # 3. Stash links BEFORE emphasis. Otherwise `__` inside a URL turns into
    # `<b>` in the href, and a `[^)]+` URL group stops at the first `)` for
    # any link with balanced inner parens. The scheme is whitelisted here
    # too, so `javascript:` etc. don't ride through as clickable.
    links: list[str] = []
    def _process_links(s: str) -> str:
        out: list[str] = []
        i = 0
        while i < len(s):
            m = re.match(r"\[([^\]\n]+)\]\(", s[i:])
            if m is None:
                out.append(s[i])
                i += 1
                continue
            disp = m.group(1)
            url_start = i + m.end()
            url_end = _extract_url(s, url_start)
            if url_end == -1:
                out.append(s[i])
                i += 1
                continue
            url = s[url_start:url_end]
            if not _is_safe_url(url):
                # Disallowed scheme: leave the literal `[text](url)` so the
                # user sees what they wrote instead of a silent drop.
                out.append(s[i])
                i += 1
                continue
            links.append(f'<a href="{_escape_html(url)}">{_escape_html(disp)}</a>')
            out.append(link_slot(len(links) - 1))
            i = url_end + 1
        return "".join(out)
    text = _process_links(text)

    # 4. Escape what survives stashing.
    text = _escape_html(text)

    # 5. Bold (** or __), then italic (* or _). Content boundary class
    # forbids whitespace AND the marker char itself, so a spurious `* a *`
    # (with internal whitespace) doesn't match, bullets at line start (`* item`)
    # don't match, and the outer pair of `** ab **` can't grab `* ab *` as its
    # italic content. Doubled markers run before single so the `*` pairs aren't
    # consumed by the italic pass.
    text = re.sub(r"\*\*([^*\s\n](?:[^*\n]*?[^*\s\n])?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![_\w])__([^_\s\n](?:[^_\n]*?[^_\s\n])?)__(?!\w)", r"<b>\1</b>", text)
    text = re.sub(r"(?<![*\w])\*([^*\s\n](?:[^*\n]*?[^*\s\n])?)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<![_\w])_([^_\s\n](?:[^_\n]*?[^_\s\n])?)_(?!\w)", r"<i>\1</i>", text)

    # 6. Strikethrough.
    text = re.sub(r"~~(\S(?:[^~\n]*?\S)?)~~", r"<s>\1</s>", text)

    # 7. Headers AFTER emphasis. Strip any inner `<b>` tags from the body
    # before wrapping, otherwise `## **Foo**` becomes `<b><b>Foo</b></b>` which
    # Telegram rejects. The header is already bold; the inner ones are dupes.
    def _wrap_header(m: re.Match) -> str:
        body = m.group(1).strip().replace("<b>", "").replace("</b>", "")
        return f"<b>{body}</b>"
    text = re.sub(r"^#{1,6}\s+(.+)$", _wrap_header, text, flags=re.MULTILINE)

    # 8. Bullet markers -> Unicode bullet, preserving indent.
    text = re.sub(r"^([ \t]*)[*-][ \t]+", r"\1• ", text, flags=re.MULTILINE)

    # 9. Re-insert stashed content. Links first because their HTML may contain
    # inline-code slots from a `[see \`foo\`](url)` pattern, and we want those
    # resolved on the next pass.
    for i, l in enumerate(links):
        text = text.replace(link_slot(i), l)
    for i, c in enumerate(inlines):
        text = text.replace(inline_slot(i), c)
    for i, c in enumerate(fences):
        text = text.replace(code_slot(i), c)

    # 10. Final balance check. Mis-nested output (interleaved span markers)
    # would get rejected by Telegram and trigger the `_send` BadRequest
    # fallback; do the same here so the failure is local and we save the
    # http round-trip.
    if not _is_balanced(text):
        return _escape_html(original)

    return text


# Used by chunk_markdown to identify fence lines (up to 3 leading spaces of
# indent are allowed per CommonMark spec).
_FENCE_RE = re.compile(r"^[ \t]{0,3}```")


def _code_mask(text: str) -> set[int]:
    """Return the set of character positions that fall inside any fenced code
    block, including the fence lines themselves. Used by chunk_markdown to
    avoid splitting in the middle of a code block (which would leave one
    chunk holding an unclosed ``` and let the body escape its stash)."""
    masked: set[int] = set()
    in_code = False
    pos = 0
    for line in text.split("\n"):
        line_end = pos + len(line)
        is_fence = bool(_FENCE_RE.match(line))
        # The fence line itself is part of the block (both opening + closing).
        in_block = in_code or is_fence
        if in_block:
            masked.update(range(pos, line_end + 1))  # +1 to cover the trailing \n
        if is_fence:
            in_code = not in_code
        pos = line_end + 1
    return masked


def chunk_markdown(text: str, limit: int) -> list[str]:
    """Split markdown into <=limit-char chunks at the latest sensible boundary
    (paragraph, then line, then space, then hard slice). Tracks fenced code
    blocks and refuses split points that land inside one when possible -- a
    split inside a fence leaves the leading chunk with an unclosed ``` and
    md_to_html can't stash it, so the body would leak. Empty chunks are
    dropped so a leading newline doesn't produce a zero-length send (which
    the Telegram API rejects)."""
    if len(text) <= limit:
        return [text] if text.strip() else []

    code_mask = _code_mask(text)
    def safe_split(cut: int) -> bool:
        # A split is safe iff the byte immediately before AND after are both
        # outside any fenced code block -- otherwise we'd cut the block.
        return (cut - 1 not in code_mask) and (cut not in code_mask)

    chunks: list[str] = []
    rest_start = 0
    while len(text) - rest_start > limit:
        window_end = rest_start + limit
        cut = -1
        # Prefer paragraph -> line -> word boundary.
        for delim, after_len in (("\n\n", 2), ("\n", 1), (" ", 1)):
            search_to = window_end - len(delim)
            while True:
                pos = text.rfind(delim, rest_start, search_to + 1)
                if pos < rest_start:
                    break
                candidate = pos + after_len
                if safe_split(candidate):
                    cut = candidate
                    break
                search_to = pos - 1
            if cut > rest_start:
                break

        if cut <= rest_start:
            # No safe boundary found; hard-cut at the limit. May leave a
            # broken code block, but `_send` catches the resulting BadRequest
            # and retries plain text for this chunk.
            cut = window_end

        chunk = text[rest_start:cut].strip()
        if chunk:
            chunks.append(chunk)
        rest_start = cut
        # Skip leading whitespace on the next chunk so we don't emit a chunk
        # that's nothing but a line break.
        while rest_start < len(text) and text[rest_start] in " \t\n":
            rest_start += 1

    tail = text[rest_start:].strip()
    if tail:
        chunks.append(tail)
    return chunks
