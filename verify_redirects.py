#!/usr/bin/env python3
"""Statically verify every redirect stub points at the correct new-site URL.

For each navigable ``*.html`` page this checks that the <meta refresh>,
<link canonical> and JS ``location.replace`` targets all agree and equal the
expected ``https://docs.sglang.io/<same path>`` (root index -> bare homepage),
and that no stale Sphinx content was left behind. Exits non-zero on any problem.

It also writes the unique target list to /tmp/redirect_targets.txt so a live
reachability check (curl) can confirm each target actually exists on the new
site.

Run from the repo root:  python3 verify_redirects.py
"""
import re
import sys
from pathlib import Path

# Reuse the exact same target logic the generator uses (single source of truth).
from make_redirects import NEW_BASE, is_asset, target_for

ROOT = Path(__file__).resolve().parent

META = re.compile(r'content="0; url=([^"]+)"')
CANON = re.compile(r'<link rel="canonical" href="([^"]+)"')
JS = re.compile(r'location\.replace\("([^"]+)"')


def main() -> None:
    problems = []
    targets = set()
    checked = 0

    for path in sorted(ROOT.rglob("*.html")):
        rel = path.relative_to(ROOT)
        if is_asset(rel) or rel.name == "404.html":
            continue
        html = path.read_text(encoding="utf-8")
        exp = target_for(rel)
        targets.add(exp)
        checked += 1

        meta = META.search(html)
        canon = CANON.search(html)
        js = JS.search(html)
        got = (
            meta.group(1) if meta else None,
            canon.group(1) if canon else None,
            js.group(1) if js else None,
        )
        if any(g != exp for g in got):
            problems.append((rel.as_posix(), f"expected {exp}, got {got}"))
        if "pydata-sphinx" in html or "searchindex" in html:
            problems.append((rel.as_posix(), "stale Sphinx content remains"))

    # 404 catch-all must forward the original path.
    nf = (ROOT / "404.html").read_text(encoding="utf-8")
    if f'"{NEW_BASE}" + location.pathname' not in nf:
        problems.append(("404.html", "does not forward location.pathname"))

    print(f"Checked {checked} stubs; {len(targets)} unique targets; 404.html present.")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for name, msg in problems:
            print(f"  - {name}: {msg}")
        sys.exit(1)

    Path("/tmp/redirect_targets.txt").write_text("\n".join(sorted(targets)) + "\n")
    print("OK: every stub's meta/canonical/JS target is consistent and correct.")
    print("Wrote unique targets to /tmp/redirect_targets.txt for the live check.")


if __name__ == "__main__":
    main()
