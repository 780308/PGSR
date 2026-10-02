"""Compare clean release source with the installed PGSR package, ignoring EOL style."""

import json
from pathlib import Path

import pgsr


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "upstream/PGSR-python/pgsr"
INSTALLED = Path(pgsr.__file__).resolve().parent


def normalized(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n")


def main() -> None:
    files = sorted(path.relative_to(SOURCE) for path in SOURCE.rglob("*.py"))
    differing = [str(rel) for rel in files if not (INSTALLED / rel).is_file() or normalized(SOURCE / rel) != normalized(INSTALLED / rel)]
    report = {
        "release_source": str(SOURCE),
        "installed_package": str(INSTALLED),
        "python_file_count": len(files),
        "differing_or_missing": differing,
    }
    print(json.dumps(report, indent=2))
    if differing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
