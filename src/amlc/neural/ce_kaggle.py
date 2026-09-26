"""Cross-encoder for pair scoring (day-2 plan, stage 4). Runs on a Kaggle GPU notebook, not locally.

Model: xlm-roberta-base (MIT licence, 278M parameters), fine-tuned only on competition train pairs.
It reads both raw records together ("name | address | country" on each side) and outputs a match
probability. It is applied only to pairs LightGBM is unsure about; a stacker fitted on VAL-A blends
the two scores, so candidate_pairs.tsv stays the LightGBM input (first scoring model).

Kaggle setup: Accelerator "GPU T4 x2" (or P100), Internet ON (to download the model once), and the
exported pair files (data/_v2/ce/*.parquet) uploaded as a private dataset. Columns expected in every
file: s1_gid, s23_gid, name_1, addr_1, country_1, name_2, addr_2, country_2, and label (train/val).

Usage in a notebook cell:
    !python ce_kaggle.py train   --data /kaggle/input/<dataset> --out /kaggle/working
    !python ce_kaggle.py predict --data /kaggle/input/<dataset> --out /kaggle/working --file val.parquet
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 96
SEED = 20260927


def side(df: pd.DataFrame, k: int) -> list[str]:
    def col(c):
        return df[f"{c}_{k}"].fillna("").astype(str)
    return (col("name") + " | " + col("addr") + " | " + col("country")).tolist()


class Pairs(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.a, self.b = side(df, 1), side(df, 2)
        self.y = df["label"].astype("float32").to_numpy() if "label" in df else None

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return i


def collate(tok, ds):
    def fn(idx):
        enc = tok([ds.a[i] for i in idx], [ds.b[i] for i in idx], truncation="longest_first",
                  max_length=MAX_LEN, padding=True, return_tensors="pt")
        if ds.y is not None:
            enc["labels"] = torch.tensor(ds.y[idx], dtype=torch.float32)
        return enc
    return fn


def load(data_dir: str, name: str, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(data_dir, name))
    if limit and len(df) > limit:
        df = df.sample(n=limit, random_state=SEED)
    return df.reset_index(drop=True)


def train(args):
    torch.manual_seed(SEED)
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(dev)
    df = load(args.data, "train.parquet", args.max_train)
    print(f"train pairs {len(df):,}  positives {int(df['label'].sum()):,}", flush=True)
    ds = Pairs(df)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate(tok, ds), num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(dl) * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    t0, step = time.time(), 0
    for _ in range(math.ceil(args.epochs)):
        for batch in dl:
            y = batch.pop("labels").to(dev)
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = lossf(model(**batch).logits.squeeze(-1), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 200 == 0:
                rate = step * args.batch / (time.time() - t0)
                print(f"step {step}/{steps} loss {loss.item():.4f} {rate:,.0f} pairs/s", flush=True)
            if step % args.save_every == 0 or step == steps:
                model.save_pretrained(os.path.join(args.out, "ce_model"))
                tok.save_pretrained(os.path.join(args.out, "ce_model"))
            if step >= steps:
                break
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


@torch.no_grad()
def predict(args):
    dev = torch.device("cuda")
    path = os.path.join(args.out, "ce_model")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(dev).half().eval()
    df = load(args.data, args.file)
    ds = Pairs(df)
    order = np.argsort([len(a) + len(b) for a, b in zip(ds.a, ds.b)])  # length-sorted batches
    probs = np.empty(len(ds), dtype=np.float32)
    t0 = time.time()
    for s in range(0, len(order), args.batch * 4):
        idx = order[s:s + args.batch * 4]
        enc = tok([ds.a[i] for i in idx], [ds.b[i] for i in idx], truncation="longest_first",
                  max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
        probs[idx] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
    print(f"scored {len(ds):,} pairs in {(time.time() - t0) / 60:.1f} min", flush=True)
    out = df[["s1_gid", "s23_gid"]].copy()
    out["p_ce"] = probs
    out.to_parquet(os.path.join(args.out, "p_ce_" + args.file), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "predict"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--file", default="val.parquet")
    ap.add_argument("--max-train", type=int, default=1_000_000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--save-every", type=int, default=2000)
    args = ap.parse_args()
    train(args) if args.mode == "train" else predict(args)


if __name__ == "__main__":
    main()
