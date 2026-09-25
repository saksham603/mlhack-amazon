"""Static scan for label shortcuts in production code (src/). Python cannot physically stop a
file read, so this is a detector, run by the test suite, not a lock.

R1: outside amlc/foundation/, no code may reference label artifacts or the leaky v1 split
    (only the foundation build scripts and access.py may).
R2: outside amlc/eval/ and amlc/foundation/access.py, no code may request VALIDATION or LOCKBOX
    label data or call the evaluation modules' private loaders.
"""
import re
from pathlib import Path

R1_PATTERN = re.compile(r"gt_links|gt_raw|s1_match_counts|labels[\\/]+v\d|splits[\\/]+v1")
R2_PATTERN = re.compile(
    r"""load_(?:labels|match_counts)\(\s*(?:split\s*=\s*)?['"](?:VALIDATION|LOCKBOX)['"]"""
    r"""|_load_(?:validation|lockbox)_[a-z_]+""",
    re.IGNORECASE,
)


def scan(src_root: Path) -> list[tuple[str, str, str]]:
    src_root = Path(src_root)
    violations = []
    for f in sorted(src_root.rglob("*.py")):
        rel = f.relative_to(src_root).as_posix()
        text = f.read_text(encoding="utf-8")
        if not rel.startswith("amlc/foundation/"):
            for m in R1_PATTERN.finditer(text):
                violations.append((rel, "R1 direct label-artifact reference", m.group(0)))
        if not (rel.startswith("amlc/eval/") or rel == "amlc/foundation/access.py"):
            for m in R2_PATTERN.finditer(text):
                violations.append((rel, "R2 restricted label access", m.group(0)))
    return violations
