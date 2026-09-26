"""Unit tests for day-2 cleanup v2 name rules (synthetic rows; no data files)."""
import polars as pl

from amlc.features import norm_v2 as N


def _names(rows):
    df = pl.DataFrame({"gid": list(range(len(rows))), "country": [r[0] for r in rows],
                       "name_tokens": [r[1] for r in rows], "name_legal_form": [r[2] for r in rows]},
                      schema={"gid": pl.UInt32, "country": pl.String, "name_tokens": pl.List(pl.String),
                              "name_legal_form": pl.String})
    return N.name_fields(df)


def test_titles_web_legal_and_stop_words():
    out = _names([("India", ["m", "s", "sharma", "traders", "pvt"], "ltd"),
                  ("US", ["www", "holderherrera", "com"], ""),
                  ("France", ["europ", "freres", "s", "a"], ""),
                  ("India", ["shri", "ram", "and", "sons"], "")])
    assert out["core"].to_list() == ["sharma traders", "holderherrera", "europ freres", "ram sons"]
    assert out["legal"].to_list() == ["ltd pvt", "", "sa", ""]


def test_indic_tokens_are_romanised_and_flagged():
    # two rows: polars treats a one-row offset column as a scalar (real batches are millions of rows)
    out = _names([("India", ["रेस्टोरेंट"], ""), ("US", ["alpha", "beta"], "")])
    assert out["indic"].to_list() == [True, False]
    assert out["core"][0].isascii() and len(out["core"][0]) > 3


def test_skeleton_maps_lookalikes_and_drops_vowels():
    df = pl.DataFrame({"ns": ["michae1ine", "crumrnie", "jeffrson"]})
    got = df.select(N.skeleton_expr(pl.col("ns")))["ns"].to_list()
    assert got[0] == "mchln" and got[1] == "crm" and got[2] == "jfrsn"


def test_state_table_maps_names_and_codes():
    t = N.state_table()
    look = {(r["country"], r["comp_key"]): r["code"] for r in t.iter_rows(named=True)}
    assert look[("US", "north carolina")] == "nc" and look[("US", "nc")] == "nc"
    assert look[("India", "maharashtra")] == "mh" and look[("India", "orissa")] == "od"
