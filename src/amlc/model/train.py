"""D-S8: LightGBM binary classifier, defaults + early stopping on a FIT holdout, fixed seed.
No hyperparameter search tonight (sprint scope).
"""
import lightgbm as lgb
import polars as pl

FEATURE_COLS = ["name_jaccard", "name_overlap_n", "addr_token_jaccard", "house_number_match",
                "zip_match", "legal_form_match", "legal_form_both_present",
                "blocking_score", "blocking_rank"]
SEED = 20260926


def to_xy(df: pl.DataFrame, feature_cols: list[str] = FEATURE_COLS):
    X = df.select(feature_cols).to_numpy()
    y = df["label"].to_numpy()
    return X, y


def train_test_split(df: pl.DataFrame, holdout_frac: float = 0.1, seed: int = SEED):
    """Split by S1 (not by row), so no S1's candidates leak across train/holdout."""
    s1_ids = df.select("s1_gid").unique().sort("s1_gid")  # sorted: unique() order is not stable
    n_holdout = max(1, round(s1_ids.height * holdout_frac))
    holdout_s1 = s1_ids.sample(n=n_holdout, seed=seed, shuffle=True, with_replacement=False)
    holdout = df.join(holdout_s1, on="s1_gid", how="semi")
    train = df.join(holdout_s1, on="s1_gid", how="anti")
    return train, holdout


def train(train_df: pl.DataFrame, feature_cols: list[str] = FEATURE_COLS, seed: int = SEED,
          holdout_frac: float = 0.1, num_boost_round: int = 500, early_stopping_rounds: int = 30):
    fit_df, holdout_df = train_test_split(train_df, holdout_frac, seed)
    Xtr, ytr = to_xy(fit_df, feature_cols)
    Xho, yho = to_xy(holdout_df, feature_cols)
    dtrain = lgb.Dataset(Xtr, label=ytr, feature_name=feature_cols)
    dholdout = lgb.Dataset(Xho, label=yho, reference=dtrain, feature_name=feature_cols)
    params = {"objective": "binary", "metric": "binary_logloss", "seed": seed, "verbosity": -1,
              "deterministic": True, "force_row_wise": True}
    booster = lgb.train(params, dtrain, num_boost_round=num_boost_round, valid_sets=[dholdout],
                        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)])
    return booster


def predict_proba(booster: lgb.Booster, df: pl.DataFrame, feature_cols: list[str] = FEATURE_COLS):
    X = df.select(feature_cols).to_numpy()
    return booster.predict(X, num_iteration=booster.best_iteration)
