from pathlib import Path

def get_readme() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "README.md").read_text(encoding="utf-8")