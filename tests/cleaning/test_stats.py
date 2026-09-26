"""C5 tests: the required gate is token-level == row-level equivalence (§3 C5)."""
import random

import polars as pl

from amlc.cleaning import stats


def _sample(n, seed=1):
    rng = random.Random(seed)
    names = ["acme traders", "beta corp", "acme llc", "gamma industries", "", "delta acme"]
    addrs = ["1 main st", "2 oak ave", "", "3 elm rd", "1 main st"]
    countries = ["US", "India"]
    return pl.DataFrame({
        "name_clean": [rng.choice(names) for _ in range(n)],
        "addr_clean": [rng.choice(addrs) for _ in range(n)],
        "country": [rng.choice(countries) for _ in range(n)],
    })


def test_token_level_matches_row_level_reference_on_a_sample():
    df = _sample(2000)
    fast = stats.document_frequency(df, "name_clean", "addr_clean", ["country"])
    ref = stats.document_frequency_row_level(df, "name_clean", "addr_clean", ["country"])
    fast_s = fast.select("country", "token", "df", "idf").sort("country", "token")
    ref_s = ref.select("country", "token", "df", "idf").sort("country", "token")
    assert fast_s["df"].to_list() == ref_s["df"].to_list()
    assert fast_s["token"].to_list() == ref_s["token"].to_list()
    for a, b in zip(fast_s["idf"].to_list(), ref_s["idf"].to_list()):
        assert abs(a - b) < 1e-9


def test_empty_strings_produce_no_token():
    df = pl.DataFrame({"name_clean": [""], "addr_clean": [""], "country": ["US"]})
    out = stats.document_frequency(df, "name_clean", "addr_clean", ["country"])
    assert out.height == 0


def test_additive_partial_counts_equal_whole_pool_computation():
    # R5/CG-21: the additive (per-table, then summed) path must equal computing on the whole pool
    # at once. Split one sample into 3 disjoint "tables" and compare against the combined pool.
    whole = _sample(1500, seed=7)
    thirds = [whole.slice(i, 500) for i in (0, 500, 1000)]
    n_parts, df_parts = [], []
    for part in thirds:
        n, d = stats.partial_counts(part, "name_clean", "addr_clean", ["country"])
        n_parts.append(n)
        df_parts.append(d)
    additive = stats.finalize_idf(n_parts, df_parts, ["country"]).select("country", "token", "df", "idf")
    direct = stats.document_frequency(whole, "name_clean", "addr_clean", ["country"]).select(
        "country", "token", "df", "idf")
    a = additive.sort("country", "token")
    b = direct.sort("country", "token")
    assert a["df"].to_list() == b["df"].to_list()
    assert a["token"].to_list() == b["token"].to_list()
    for x, y in zip(a["idf"].to_list(), b["idf"].to_list()):
        assert abs(x - y) < 1e-9


def test_df_counts_documents_not_occurrences():
    # "acme" appears twice in one row's name+addr tokens combined logically once per doc (list.unique)
    df = pl.DataFrame({"name_clean": ["acme acme"], "addr_clean": [""], "country": ["US"]})
    out = stats.document_frequency(df, "name_clean", "addr_clean", ["country"])
    assert out.filter(pl.col("token") == "acme")["df"].to_list() == [1]
