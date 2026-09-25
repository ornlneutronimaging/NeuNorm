"""Re-download the intersphinx inventories committed next to this script.

Run with ``pixi run update-inventories``. The projects and URLs come from
``intersphinx_mapping`` in ``docs/conf.py``: each project whose locations include a
local ``_inventory/<name>.inv`` fallback gets that file refreshed from
``<url>/objects.inv``. Uses only the standard library, so it needs no extra tools.
"""

import runpy
import sys
import urllib.request
import zlib
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent
TIMEOUT = 60
# Some sites (numpy.org) reject the default "Python-urllib" user agent with a 403.
USER_AGENT = "NeuNorm-docs-update-inventories (+https://github.com/ornlneutronimaging/NeuNorm)"


def fetch(url: str) -> bytes:
    """Download ``url`` and check it is a Sphinx v2 inventory before returning it."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        data = response.read()
    header, sep, rest = data.partition(b"\n")
    if header != b"# Sphinx inventory version 2" or not sep:
        raise ValueError(f"{url} did not return a Sphinx v2 inventory")
    # The body after the four header lines is zlib-compressed; make sure it decompresses.
    zlib.decompress(rest.split(b"\n", 3)[3])
    return data


def main() -> int:
    """Refresh every committed inventory named in ``intersphinx_mapping``."""
    mapping = runpy.run_path(str(DOCS_DIR / "conf.py"))["intersphinx_mapping"]
    failed = []
    for name, (uri, locations) in mapping.items():
        local = [loc for loc in locations if loc and "://" not in loc]
        if not local:
            continue
        url = f"{uri.rstrip('/')}/objects.inv"
        try:
            data = fetch(url)
        except Exception as err:  # noqa: BLE001 - report every failure, then keep going
            print(f"{name}: FAILED to fetch {url}: {err}", file=sys.stderr)
            failed.append(name)
            continue
        for loc in local:
            target = DOCS_DIR / loc
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(target)
            print(f"{name}: {url} -> {target.relative_to(DOCS_DIR.parent)} ({len(data)} bytes)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
