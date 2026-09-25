"""The label-shortcut scanner finds nothing in real code, and does catch planted violations."""
from pathlib import Path

from amlc.foundation import leakscan

SRC = Path(r"C:\Users\suremdra singh\amlc2026\src")


def _plant(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_real_source_tree_is_clean():
    assert leakscan.scan(SRC) == []


def test_detects_direct_label_file_read(tmp_path):
    _plant(tmp_path, "amlc/cleaning/bad.py", 'pl.read_parquet("data/labels/v1/" + "gt_links.parquet")\n')
    rules = {v[1] for v in leakscan.scan(tmp_path)}
    assert any(r.startswith("R1") for r in rules)


def test_detects_leaky_v1_split_read(tmp_path):
    _plant(tmp_path, "amlc/features/bad.py", 'p = "data/splits/v1/s1_split.parquet"\n')
    assert any(v[1].startswith("R1") for v in leakscan.scan(tmp_path))


def test_detects_validation_label_request(tmp_path):
    _plant(tmp_path, "amlc/model/bad.py", "access.load_labels('VALIDATION')\naccess.load_match_counts(split=\"LOCKBOX\")\n")
    assert sum(v[1].startswith("R2") for v in leakscan.scan(tmp_path)) == 2


def test_detects_private_scorer_loader_call(tmp_path):
    _plant(tmp_path, "amlc/model/bad.py", "scorer._load_validation_labels()\n")
    assert any(v[1].startswith("R2") for v in leakscan.scan(tmp_path))


def test_allows_fit_labels_and_foundation_builders(tmp_path):
    _plant(tmp_path, "amlc/cleaning/ok.py", "access.load_labels('FIT')\n")
    _plant(tmp_path, "amlc/foundation/builder.py", 'x = "gt_links.parquet"\n')
    assert leakscan.scan(tmp_path) == []
