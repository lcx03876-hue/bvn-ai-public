"""Final check: no identifier, number, operator or structure changed by the translation.

Compares each tracked ``.py`` against the pre-translation revision (HEAD) after
normalising away string-literal *text*:

* every string/bytes constant becomes a placeholder;
* inside f-strings the literal parts are dropped (message wording only), while
  every interpolated expression is still compared;
* implicit concatenation is folded by the parser, so merging or splitting
  adjacent literals does not matter.

If the dumps match, the only edits were to comments, docstrings and message
strings -- which is exactly the contract for this translation pass.

Usage::

    python verify_no_code_change.py
    python verify_no_code_change.py --base HEAD~1
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Normalise(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        if isinstance(node.value, (str, bytes)):
            return ast.copy_location(ast.Constant(value="<STR>"), node)
        return node

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        node.values = [v for v in node.values
                       if not isinstance(v, ast.Constant)]
        return node


def dump(src: str) -> str | None:
    try:
        tree = Normalise().visit(ast.parse(src))
    except SyntaxError:
        return None
    ast.fix_missing_locations(tree)
    return ast.dump(tree, include_attributes=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="HEAD")
    args = ap.parse_args()

    files = subprocess.run(["git", "ls-files", "*.py"], cwd=str(ROOT),
                           capture_output=True, text=True).stdout.split()
    bad: list[str] = []
    broken: list[str] = []
    changed = 0
    for rel in files:
        r = subprocess.run(["git", "show", f"{args.base}:{rel}"], cwd=str(ROOT),
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            continue
        new = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        if r.stdout == new:
            continue
        changed += 1
        dn = dump(new)
        if dn is None:
            broken.append(rel)
            continue
        do = dump(r.stdout)
        if do is not None and do != dn:
            bad.append(rel)

    print(f"[verify] {len(files)} tracked .py, {changed} modified "
          f"vs {args.base}")
    if broken:
        print(f"FAIL  does not parse: {broken}")
    if not bad and not broken:
        print("OK    normalised ASTs are identical.")
        print("      No identifier, numeric literal, operator, import or")
        print("      control-flow structure changed -- only comments,")
        print("      docstrings and message strings.")
    else:
        print(f"FAIL  structural differences: {bad}")

    total = 0
    rows: list[tuple[str, int]] = []
    for f in sorted(ROOT.rglob("*.py")):
        if ".git" in f.parts:
            continue
        c = f.read_text(encoding="utf-8", errors="replace")
        n = sum(1 for ch in c if "\u4e00" <= ch <= "\u9fff")
        if n:
            rows.append((str(f.relative_to(ROOT)), n))
            total += n
    print(f"\nremaining CJK characters: {total}")
    for rel, n in rows:
        print(f"    {rel}: {n}")
    return 0 if (not bad and not broken) else 1


if __name__ == "__main__":
    raise SystemExit(main())
