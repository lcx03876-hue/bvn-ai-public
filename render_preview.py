"""Render the repository's Markdown docs as a local GitHub-style website.

No Markdown library is installed, so this implements the subset the docs
actually use: ATX headings, fenced code blocks, tables, ordered and unordered
lists, blockquotes, horizontal rules, and inline code / bold / italic / links.

Output is a single self-contained directory with an ``index.html`` plus one
page per document and a shared stylesheet that imitates GitHub's light theme,
so the public README can be reviewed without pushing anything.

Usage::

    python render_preview.py            # build and print the entry path
    python render_preview.py --open     # also open it in the default browser
"""

from __future__ import annotations

import argparse
import html
import re
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "preview"

# Documents to render, in navigation order.
DOCS = [
    ("README.md", "index.html", "Overview"),
    ("docs/GETTING_STARTED.md", "getting-started.html", "Getting started"),
    ("docs/ARCHITECTURE.md", "architecture.html", "Architecture"),
    ("docs/RESULTS.md", "results.html", "Results"),
    ("game_patches/README.md", "game-patches.html", "game_patches"),
    ("NOTICE", "notice.html", "Notice"),
]

INLINE_CODE = re.compile(r"`([^`]+)`")
# Bold is matched before italic and its body may contain single
# asterisks, so that **... *emphasis* ...** works.
BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def inline(text: str) -> str:
    """Render inline Markdown. Code spans are protected from other rules."""
    held: list[str] = []

    def stash(m: re.Match) -> str:
        held.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00{len(held) - 1}\x00"

    text = INLINE_CODE.sub(stash, text)
    text = html.escape(text)
    text = LINK.sub(
        lambda m: f'<a href="{html.escape(m.group(2))}">{m.group(1)}</a>', text)
    text = BOLD.sub(r"<strong>\1</strong>", text)
    text = ITALIC.sub(r"<em>\1</em>", text)
    for i, code in enumerate(held):
        text = text.replace(f"\x00{i}\x00", code)
    return text


def slug(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"\s+", "-", s).strip("-")


def render(md: str) -> tuple[str, list[tuple[str, str]]]:
    """Convert Markdown to HTML. Returns (html, [(heading_id, title), ...])."""
    lines = md.splitlines()
    out: list[str] = []
    toc: list[tuple[str, str]] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # fenced code block (the fence may be indented, e.g. inside a list)
        if line.lstrip().startswith("```"):
            lang = line.lstrip()[3:].strip()
            i += 1
            buf = []
            while i < n and not lines[i].lstrip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            cls = f' class="language-{lang}"' if lang else ""
            out.append(f"<pre><code{cls}>"
                       + html.escape("\n".join(buf)) + "</code></pre>")
            continue

        # table
        if (line.startswith("|") and i + 1 < n
                and re.match(r"^\|[\s:|-]+\|$", lines[i + 1])):
            header = [c.strip() for c in line.strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            t = ["<table><thead><tr>"]
            t += [f"<th>{inline(c)}</th>" for c in header]
            t.append("</tr></thead><tbody>")
            for r in rows:
                t.append("<tr>" + "".join(
                    f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table>")
            out.append("".join(t))
            continue

        # headings
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            lvl = len(m.group(1))
            title = m.group(2).strip()
            sid = slug(re.sub(r"[`*]", "", title))
            if lvl in (2, 3):
                toc.append((sid, re.sub(r"[`*]", "", title)))
            out.append(f'<h{lvl} id="{sid}">{inline(title)}</h{lvl}>')
            i += 1
            continue

        # horizontal rule
        if re.match(r"^-{3,}$", line.strip()):
            out.append("<hr>")
            i += 1
            continue

        # blockquote
        if line.startswith(">"):
            buf = []
            while i < n and lines[i].startswith(">"):
                buf.append(lines[i].lstrip("> ").rstrip())
                i += 1
            out.append("<blockquote>" + inline(" ".join(buf)) + "</blockquote>")
            continue

        # lists
        if re.match(r"^\s*[-*]\s+", line):
            buf = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                buf.append(re.sub(r"^\s*[-*]\s+", "", lines[i]))
                i += 1
            out.append("<ul>" + "".join(
                f"<li>{inline(b)}</li>" for b in buf) + "</ul>")
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            buf = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                buf.append(re.sub(r"^\s*\d+\.\s+", "", lines[i]))
                i += 1
            out.append("<ol>" + "".join(
                f"<li>{inline(b)}</li>" for b in buf) + "</ol>")
            continue

        if not line.strip():
            i += 1
            continue

        # paragraph (join consecutive non-blank lines)
        buf = []
        while (i < n and lines[i].strip()
               and not re.match(
                   r"^(#{1,6}\s|\s*```|>|\s*[-*]\s|\s*\d+\.\s|\|)", lines[i])
               and not re.match(r"^-{3,}$", lines[i].strip())):
            buf.append(lines[i].strip())
            i += 1
        out.append("<p>" + inline(" ".join(buf)) + "</p>")

    return "\n".join(out), toc


CSS = """
:root{--fg:#1f2328;--muted:#59636e;--border:#d1d9e0;--bg:#fff;--bg2:#f6f8fa;
--link:#0969da;--code-bg:#818b981f}
*{box-sizing:border-box}
body{margin:0;font:16px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',
'Noto Sans',Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg)}
header{background:#24292f;color:#fff;padding:14px 24px;display:flex;
align-items:center;gap:18px;flex-wrap:wrap}
header .repo{font-weight:600}
header a{color:#fff;text-decoration:none;opacity:.85;font-size:14px}
header a:hover{opacity:1;text-decoration:underline}
header a.on{opacity:1;font-weight:600;border-bottom:2px solid #fd8c73}
.wrap{max-width:1020px;margin:0 auto;padding:28px 24px 80px}
h1,h2,h3{line-height:1.25;margin:26px 0 14px;font-weight:600}
h1{font-size:2em;border-bottom:1px solid var(--border);padding-bottom:.3em}
h2{font-size:1.5em;border-bottom:1px solid var(--border);padding-bottom:.3em}
h3{font-size:1.2em}
a{color:var(--link);text-decoration:none}
a:hover{text-decoration:underline}
code{background:var(--code-bg);padding:.2em .4em;border-radius:6px;
font:85%/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}
pre{background:var(--bg2);padding:16px;border-radius:6px;overflow:auto}
pre code{background:none;padding:0;font-size:85%}
table{border-collapse:collapse;margin:16px 0;display:block;overflow:auto}
th,td{border:1px solid var(--border);padding:6px 13px}
th{background:var(--bg2);font-weight:600}
tr:nth-child(2n) td{background:#f6f8fa}
blockquote{border-left:4px solid var(--border);color:var(--muted);
margin:0 0 16px;padding:0 1em}
hr{border:0;border-top:1px solid var(--border);margin:24px 0}
ul,ol{padding-left:2em}
li{margin:.25em 0}
.toc{background:var(--bg2);border:1px solid var(--border);border-radius:6px;
padding:12px 18px;margin:0 0 26px;font-size:14px}
.toc b{display:block;margin-bottom:6px}
.toc a{display:block;padding:1px 0;color:var(--muted)}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    (OUT / "style.css").write_text(CSS, encoding="utf-8")

    nav = "".join(
        f'<a href="{dst}">{label}</a>' for _, dst, label in DOCS)

    built = []
    for src_rel, dst, label in DOCS:
        src = ROOT / src_rel
        if not src.exists():
            print(f"  skip {src_rel} (missing)")
            continue
        body, toc = render(src.read_text(encoding="utf-8"))
        toc_html = ""
        if toc:
            toc_html = ('<div class="toc"><b>On this page</b>'
                        + "".join(f'<a href="#{i}">{t}</a>' for i, t in toc[:14])
                        + "</div>")
        page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(label)} - BVN-AI</title>
<link rel="stylesheet" href="style.css"></head><body>
<header><span class="repo">BVN-AI</span>{nav}</header>
<div class="wrap">{toc_html}{body}</div>
</body></html>"""
        (OUT / dst).write_text(page, encoding="utf-8")
        built.append(dst)
        print(f"  {src_rel:<28} -> preview/{dst}")

    entry = (OUT / "index.html").resolve()
    print(f"\n[preview] {len(built)} pages -> {entry}")
    print(f"[preview] open: file:///{str(entry).replace(chr(92), '/')}")
    if args.open:
        webbrowser.open(entry.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
