import polars as pl

from amlc.blocking import v2
from amlc.pipeline import fused


def _side(id_col, gids, names, addrs=None, house=None, zips=None, legal=None):
    n = len(gids)
    return pl.DataFrame({
        id_col: pl.Series(gids, dtype=pl.UInt32),
        "name_tokens": pl.Series([nm.split(" ") if nm else [] for nm in names], dtype=pl.List(pl.String)),
        "addr_components": pl.Series(addrs or [[] for _ in range(n)], dtype=pl.List(pl.String)),
        "addr_house_number": house or ["" for _ in range(n)],
        "addr_zip": zips or ["" for _ in range(n)],
        "name_legal_form": legal or ["" for _ in range(n)],
    })


def test_score_and_rank_batch_sums_name_and_addr_scores():
    # s1=1 shares a rare name token AND a house-number key with s23=10; only the name token with s23=11
    s1 = _side("s1_gid", [1], ["acme traders"], addrs=[["1 main st"]], house=["1"])
    s23 = _side("s23_gid", [10, 11],
                ["acme traders", "acme traders"],
                addrs=[["1 main st"], ["9 other rd"]],
                house=["1", "9"])
    kinds = v2.KINDS_NAME + v2.KINDS_ADDR
    k1 = v2.build_keys(s1.drop("name_legal_form"), "s1_gid", kinds)
    k23 = v2.build_keys(s23.drop("name_legal_form"), "s23_gid", kinds)
    rare = v2.idf_table(k23, s23.height, 5000)

    out = fused.score_and_rank_batch(k1, k23, rare, budget_rows=10_000, k_name=10, k_addr=10)
    row10 = out.filter(pl.col("s23_gid") == 10).row(0, named=True)
    row11 = out.filter(pl.col("s23_gid") == 11).row(0, named=True)
    assert row10["blocking_score"] > row11["blocking_score"]  # shares both name and address keys
    assert row10["blocking_rank"] == 1
    assert row11["blocking_rank"] == 2


def test_run_batch_writes_candidates_and_features(tmp_path):
    s1 = _side("s1_gid", [1], ["acme traders"], addrs=[["1 main st"]], house=["1"])
    s23 = _side("s23_gid", [10, 11], ["acme traders", "other biz"], addrs=[["1 main st"], ["2 x rd"]],
                house=["1", "2"])
    kinds = v2.KINDS_NAME + v2.KINDS_ADDR
    k23 = v2.build_keys(s23.drop("name_legal_form"), "s23_gid", kinds)
    rare = v2.idf_table(k23, s23.height, 5000)

    cand_dir, feat_dir = tmp_path / "candidates", tmp_path / "features"
    rep = fused.run_batch(s1, s23, k23, rare, budget_rows=10_000, candidates_dir=cand_dir,
                           features_dir=feat_dir, batch_idx=0)
    cand = pl.read_parquet(cand_dir / "candidates_00000.parquet")
    feat = pl.read_parquet(feat_dir / "features_00000.parquet")
    assert rep["n_s1"] == 1
    assert cand.height == rep["n_candidates"] > 0
    assert feat.height == cand.height
    assert set(cand.columns) == {"s1_gid", "s23_gid", "blocking_score", "blocking_rank"}
    assert feat.filter(pl.col("s23_gid") == 10)["name_jaccard"].item() == 1.0


def test_run_country_processes_all_s1_across_batches(tmp_path):
    n = 7
    s1 = _side("s1_gid", list(range(n)), [f"name{i}" for i in range(n)])
    s23 = _side("s23_gid", [100 + i for i in range(n)], [f"name{i}" for i in range(n)])
    cand_dir, feat_dir = tmp_path / "candidates", tmp_path / "features"
    rep = fused.run_country(s1, s23, budget_rows=10_000, batch_size=3, candidates_dir=cand_dir,
                             features_dir=feat_dir)
    assert rep["n_s1"] == n
    assert rep["n_batches"] == 3  # ceil(7/3)
    written_cand = sorted(cand_dir.glob("candidates_*.parquet"))
    written_feat = sorted(feat_dir.glob("features_*.parquet"))
    assert len(written_cand) == len(written_feat) == 3
    s1_seen = set()
    for f in written_cand:
        s1_seen |= set(pl.read_parquet(f)["s1_gid"].to_list())
    assert s1_seen == set(range(n))  # every S1 matched itself in s23 (identical names) -> all present


def test_build_country_index_reused_across_two_runs(tmp_path):
    n = 4
    s1 = _side("s1_gid", list(range(n)), [f"name{i}" for i in range(n)])
    s23 = _side("s23_gid", [100 + i for i in range(n)], [f"name{i}" for i in range(n)])
    k23, rare = fused.build_country_index(s23)

    r1 = fused.run_country_with_index(s1, s23, k23, rare, budget_rows=10_000, batch_size=2,
                                       candidates_dir=tmp_path / "c1", features_dir=tmp_path / "f1")
    r2 = fused.run_country_with_index(s1, s23, k23, rare, budget_rows=10_000, batch_size=2,
                                       candidates_dir=tmp_path / "c2", features_dir=tmp_path / "f2")
    assert r1["n_candidates_total"] == r2["n_candidates_total"] > 0  # same index -> same result


def test_run_batch_skips_when_output_already_exists(tmp_path):
    n = 2
    s1 = _side("s1_gid", list(range(n)), [f"name{i}" for i in range(n)])
    s23 = _side("s23_gid", [100 + i for i in range(n)], [f"name{i}" for i in range(n)])
    k23, rare = fused.build_country_index(s23)
    cand_dir, feat_dir = tmp_path / "candidates", tmp_path / "features"

    first = fused.run_batch(s1, s23, k23, rare, 10_000, cand_dir, feat_dir, batch_idx=0)
    assert first["skipped"] is False
    second = fused.run_batch(s1, s23, k23, rare, 10_000, cand_dir, feat_dir, batch_idx=0)
    assert second["skipped"] is True
    assert second["n_candidates"] == first["n_candidates"]
