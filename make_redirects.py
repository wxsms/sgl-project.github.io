#!/usr/bin/env python3
"""Turn this deprecated GitHub Pages site into redirect stubs.

Every navigable ``*.html`` page is rewritten into a tiny stub that forwards the
visitor to the same path on the new canonical docs site, preserving deep links,
query strings and anchor fragments. A root ``404.html`` catch-all forwards any
path that no longer exists as a file.

The new site (docs.sglang.io) is the same Sphinx build as this one, so the path
maps 1:1 (including the ``.html`` extension) -- e.g.
``/advanced_features/server_arguments.html`` ->
``https://docs.sglang.io/advanced_features/server_arguments.html``.

Run from the repo root:  python3 make_redirects.py
"""
import json
from pathlib import Path

# New canonical docs site, no trailing slash.
NEW_BASE = "https://docs.sglang.io"

ROOT = Path(__file__).resolve().parent

# Directories holding Sphinx build assets, not navigable pages. Their .html
# files (theme macros) are skipped.
SKIP_DIRS = {"_static", "_sources", "_downloads", "_images", ".git"}

# Old pages with no same-path counterpart on the new site (verified at
# deprecation time via verify_redirects.py + a live HTTP check). Forward them to
# the closest live page so no old link ever lands on a 404 on the new site.
OVERRIDES = {
    # Release-lookup tool now lives under references/ on the new site.
    "release_lookup/index.html": "/references/release_lookup.html",
    "release_lookup/README.html": "/references/release_lookup.html",
    # No counterpart on the new site -> fall back to the docs homepage.
    "README.html": "/",
    "performance_dashboard/README.html": "/",
    "genindex.html": "/",
    "search.html": "/",
    "basic_usage/hy3_preview.html": "/",
    "developer_guide/msprobe_debugging_guide.html": "/",
    "diffusion/performance/deployment_cookbook.html": "/",
}

STUB = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Page moved — SGLang Documentation</title>
<link rel="canonical" href="{target}">
<meta http-equiv="refresh" content="0; url={target}">
<script>
  // Forward to the same page on the new docs site, keeping ?query and #anchor.
  location.replace({target_js} + location.search + location.hash);
</script>
</head>
<body>
<p>This page has moved to <a href="{target}">{target}</a>. Redirecting…</p>
</body>
</html>
"""

NOT_FOUND = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Page moved — SGLang Documentation</title>
<link rel="canonical" href="{base}/">
<meta http-equiv="refresh" content="0; url={base}/">
<script>
  // Forward any unknown old URL to the same path on the new docs site.
  location.replace({base_js} + location.pathname + location.search + location.hash);
</script>
</head>
<body>
<p>This documentation has moved to <a href="{base}/">{base}</a>. Redirecting…</p>
</body>
</html>
"""


def is_asset(rel: Path) -> bool:
    return any(part in SKIP_DIRS for part in rel.parts)


def target_for(rel: Path) -> str:
    """New-site URL for a page: an override, the homepage, or the same path."""
    key = rel.as_posix()
    if key in OVERRIDES:
        return NEW_BASE + OVERRIDES[key]
    if key == "index.html":  # root homepage
        return f"{NEW_BASE}/"
    return f"{NEW_BASE}/{key}"


def main() -> None:
    count = 0
    for path in sorted(ROOT.rglob("*.html")):
        rel = path.relative_to(ROOT)
        if is_asset(rel) or rel.name == "404.html":
            continue
        target = target_for(rel)
        path.write_text(
            STUB.format(target=target, target_js=json.dumps(target)),
            encoding="utf-8",
        )
        count += 1

    (ROOT / "404.html").write_text(
        NOT_FOUND.format(base=NEW_BASE, base_js=json.dumps(NEW_BASE)),
        encoding="utf-8",
    )
    print(f"Rewrote {count} pages into redirect stubs and wrote 404.html -> {NEW_BASE}")


if __name__ == "__main__":
    main()
