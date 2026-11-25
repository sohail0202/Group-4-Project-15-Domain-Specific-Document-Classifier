"""Package the runnable project, experiment and submission files."""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELIVERABLES = ROOT / "deliverables"
FILES = (
    "START-HERE.txt", "README.md", "plan.md", "requirements.txt", "start.cmd", "app.py", "ledger.py",
    "collect.py", "train.py", "Project_15_Analysis.ipynb",
    "deliverables/Project 15 - Ledger.pptx", "deliverables/Project 15 - Technical Report.pdf",
)
DIRECTORIES = ("analysis", "data", "examples", "model", "screenshots", "tools", "web")


def build_archive():
    paths = [ROOT / file for file in FILES]
    for directory in DIRECTORIES:
        paths.extend(
            path for path in sorted((ROOT / directory).rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        )

    destination = DELIVERABLES / "Project 15 - Ledger Submission.zip"
    candidate = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(candidate, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            archive.write(path, "Ledger - Project 15/" + path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(candidate) as archive:
        damaged = archive.testzip()
        if damaged or len(archive.namelist()) != len(paths):
            raise RuntimeError(f"Could not verify the project archive: {damaged}")
    candidate.replace(destination)
    print(f"Verified {len(paths)} entries in {destination.name} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build_archive()
