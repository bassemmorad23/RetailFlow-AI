"""
One-off: convert Pydantic v1 `class Config:` blocks to v2 `model_config = ConfigDict(...)`.
Writes a .bak backup next to each changed file. Reports anything it can't convert safely.
Usage: python -m scripts.migrate_pydantic_config
"""

import pathlib
import re

# v1 option names that changed in v2 -> must be converted by hand
RENAMED = {"orm_mode", "allow_population_by_field_name", "anystr_strip_whitespace",
           "min_anystr_length", "max_anystr_length", "schema_extra", "validate_all", "fields"}

BLOCK = re.compile(r"\n(?P<indent>[ ]+)class Config:\n(?P<body>(?:(?P=indent)[ ]{4}\w+ = .+\n)+)")

for path in pathlib.Path("app").rglob("*.py"):
    src = path.read_text(encoding="utf-8")
    if "class Config:" not in src:
        continue

    skipped = []

    def convert(m):
        items = re.findall(r"(\w+) = (.+)", m.group("body"))
        if any(k in RENAMED for k, _ in items):
            skipped.append([k for k, _ in items])
            return m.group(0)
        args = ", ".join(f"{k}={v.strip()}" for k, v in items)
        return f"\n{m.group('indent')}model_config = ConfigDict({args})\n"

    new, count = BLOCK.subn(convert, src)
    if count == 0:
        continue

    if "ConfigDict" not in src:
        new = new.replace("from pydantic import ", "from pydantic import ConfigDict, ", 1)

    path.with_suffix(".py.bak").write_text(src, encoding="utf-8")
    path.write_text(new, encoding="utf-8")
    print(f"{path}: converted {count - len(skipped)}")
    for keys in skipped:
        print(f"  !! needs manual conversion: {keys}")

remaining = [str(p) for p in pathlib.Path("app").rglob("*.py") if "class Config:" in p.read_text(encoding="utf-8")]
print("remaining class Config in:", remaining or "none")