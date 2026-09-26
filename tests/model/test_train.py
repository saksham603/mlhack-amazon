import polars as pl

from amlc.model import train as T


def _synthetic(n_s1=60, cands_per_s1=5, seed=1):
    import random
    rng = random.Random(seed)
    rows = []
    for s1 in range(n_s1):
        true_idx = rng.randrange(cands_per_s1)
        for c in range(cands_per_s1):
            is_true = c == true_idx
            rows.append({
                "s1_gid": s1, "s23_gid": s1 * 100 + c,
                "name_jaccard": rng.uniform(0.8, 1.0) if is_true else rng.uniform(0.0, 0.3),
                "name_overlap_n": 5 if is_true else rng.randint(0, 1),
                "addr_token_jaccard": rng.uniform(0.7, 1.0) if is_true else rng.uniform(0.0, 0.3),
                "house_number_match": is_true,
                "zip_match": is_true,
                "legal_form_match": False, "legal_form_both_present": False,
                "blocking_score": rng.uniform(5, 10) if is_true else rng.uniform(0, 3),
                "blocking_rank": 1 if is_true else rng.randint(2, cands_per_s1),
                "label": int(is_true),
            })
    return pl.DataFrame(rows)


def test_train_test_split_keeps_s1_disjoint():
    df = _synthetic(n_s1=40)
    fit_df, holdout_df = T.train_test_split(df, holdout_frac=0.25, seed=5)
    fit_s1 = set(fit_df["s1_gid"].unique().to_list())
    holdout_s1 = set(holdout_df["s1_gid"].unique().to_list())
    assert fit_s1.isdisjoint(holdout_s1)
    assert fit_s1 | holdout_s1 == set(df["s1_gid"].unique().to_list())


def test_train_and_predict_smoke_separates_true_from_false():
    df = _synthetic(n_s1=80, seed=7)
    booster = T.train(df, holdout_frac=0.2, num_boost_round=50, early_stopping_rounds=10)
    probs = T.predict_proba(booster, df)
    df2 = df.with_columns(pl.Series("prob", probs))
    mean_true = df2.filter(pl.col("label") == 1)["prob"].mean()
    mean_false = df2.filter(pl.col("label") == 0)["prob"].mean()
    assert mean_true > mean_false + 0.3  # clearly separable synthetic signal -> model must learn it
