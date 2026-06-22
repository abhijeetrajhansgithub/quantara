from pathlib import Path


def get_readme() -> str:
    return Path("README.md").read_text(encoding="utf-8")
