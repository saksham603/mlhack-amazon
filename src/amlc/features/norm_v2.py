"""Day-2 text cleanup v2 (master prompt Phase A1). Builds one row of comparison-ready fields per
record from silver v1 (read-only) plus bronze country, into data/_v2/records/. Label-free except
the native-script state map, which is learned from FIT pairs only.

Name fields: core (name tokens, non-ASCII tokens romanised with anyascii, web prefixes/suffixes,
leading titles, legal-form words and stop words removed), ns (core without spaces), skel (ns with
look-alike characters mapped, vowels after the first character dropped, repeats collapsed), ini
(initials), first/last token, ntok, legal (canonical legal-form words), indic (name had
Indian-script characters).
Address fields: addr (canonical words of every component except the detected state), street (the
first such component), region (state code from a small hand-typed US/India list, or the learned
native-script map, else ""), lastc (last component in ASCII), hn (house number without leading
zeros), hn_digits (its digits sorted), zip.
Ambiguity fields, relative to each (dataset, country) pool: s1_share / s23_share = share of that
pool's S1 / S2+S3 records with the same core name.

Run: PYTHONPATH=src .venv/Scripts/python.exe -m amlc.features.norm_v2
"""
import json
import re
import time

import polars as pl
from anyascii import anyascii

from amlc.blocking import candidates as C
from amlc.foundation import access

OUT = access.DATA / "_v2" / "records"
SOURCES = [("train", 1), ("train", 2), ("train", 3), ("test", 1), ("test", 2), ("test", 3)]
INDIC = r"[ऀ-ൿ]"
NON_ASCII = r"[^\x00-\x7F]"

WEB_LEAD = ["www", "http", "https"]
WEB_MARK = ["com", "net", "org", "biz", "info"]
WEB_TAIL = ["com", "net", "org", "biz", "info", "co", "in", "fr"]
TITLES = ["mr", "mrs", "ms", "dr", "shri", "sri", "shree", "smt", "messrs", "formerly"]
LEGAL_CANON = {"limited": "ltd", "private": "pvt", "corporation": "corp", "company": "co",
               "incorporated": "inc"}
LEGAL = sorted({"inc", "llc", "corp", "co", "ltd", "pvt", "llp", "plc", "pllc", "lp", "sarl", "sas",
                "sasu", "eurl", "sa", "sci", "snc", "selarl", "eirl", "ei", "gmbh"} | set(LEGAL_CANON))
STOP = ["and", "the", "et", "of", "de", "la", "le", "les", "du", "des", "d", "l"]

STREET = {"road": "rd", "street": "st", "avenue": "ave", "av": "ave", "avn": "ave", "boulevard": "blvd",
          "bd": "blvd", "bvd": "blvd", "drive": "dr", "lane": "ln", "court": "ct", "place": "pl",
          "suite": "ste", "apartment": "apt", "floor": "fl", "flr": "fl", "highway": "hwy",
          "parkway": "pkwy", "circle": "cir", "terrace": "ter", "north": "n", "south": "s", "east": "e",
          "west": "w", "building": "bldg", "number": "no", "nagar": "ngr", "colony": "clny",
          "sector": "sec", "phase": "ph", "block": "blk", "opposite": "opp", "near": "nr",
          "chemin": "ch", "route": "rte", "impasse": "imp", "allee": "all", "faubourg": "fbg",
          "square": "sq", "mount": "mt", "center": "ctr", "centre": "ctr", "saint": "st"}
# Country-keyed overrides (country stays an open set: unknown countries just get the common map).
STREET_BY_COUNTRY = {"France": {"r": "rue", "st": "saint", "ste": "sainte", "saint": "saint"}}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr"}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "ct": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "kerala": "kl",
    "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz",
    "nagaland": "nl", "odisha": "od", "orissa": "od", "or": "od", "punjab": "pb", "rajasthan": "rj",
    "sikkim": "sk", "tamil nadu": "tn", "telangana": "ts", "tg": "ts", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "ua": "uk", "west bengal": "wb",
    "andaman and nicobar islands": "an", "chandigarh": "ch", "dadra and nagar haveli": "dn",
    "daman and diu": "dn", "dadra and nagar haveli and daman and diu": "dn", "dd": "dn",
    "delhi": "dl", "nct of delhi": "dl", "jammu and kashmir": "jk", "ladakh": "la",
    "lakshadweep": "ld", "puducherry": "py", "pondicherry": "py"}
STATE_TABLES = {"US": US_STATES, "India": IN_STATES}
NATIVE_MIN_SUPPORT = 20
NATIVE_MIN_SHARE = 0.9


def state_table() -> pl.DataFrame:
    rows = []
    for country, table in STATE_TABLES.items():
        for name, code in table.items():
            rows.append((country, name, code))
            rows.append((country, code, code))
    return pl.DataFrame(rows, schema=["country", "comp_key", "code"], orient="row").unique(["country", "comp_key"])


def romanise(values: pl.Series) -> pl.DataFrame:
    """anyascii for the distinct non-ASCII strings only: (k, v) with v lowercase a-z0-9 and spaces."""
    u = values.drop_nulls().unique()
    u = u.filter(u.str.contains(NON_ASCII))
    v = [re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", anyascii(s).lower())).strip() for s in u.to_list()]
    return pl.DataFrame({"k": u, "v": pl.Series(v, dtype=pl.String)})


def _collapse_repeats(e: pl.Expr) -> pl.Expr:
    for ch in "abcdefghijklmnopqrstuvwxyz0123456789":
        e = e.str.replace_all(ch + ch + "+", ch)
    return e


def skeleton_expr(ns: pl.Expr) -> pl.Expr:
    s = ns
    for a, b in (("rn", "m"), ("vv", "w"), ("ph", "f"), ("0", "o"), ("1", "l"), ("5", "s")):
        s = s.str.replace_all(a, b, literal=True)
    s = s.str.slice(0, 1) + s.str.slice(1).str.replace_all("[aeiouy]", "")
    return _collapse_repeats(s)


def name_fields(s: pl.DataFrame) -> pl.DataFrame:
    """s: gid, country, name_tokens, name_legal_form -> name fields."""
    rom = romanise(s.select(pl.col("name_tokens").explode())["name_tokens"])
    t = pl.col("name_tokens").list.eval(
        pl.element().replace(rom["k"], rom["v"]).str.replace_all(" ", "", literal=True)
        .filter(pl.element() != ""))
    df = s.select("gid", "country", t.alias("t"), pl.col("name_legal_form").fill_null(""),
                  pl.col("name_tokens").list.join(" ").str.contains(INDIC).alias("indic"))
    had_web = (pl.col("t").list.first().is_in(WEB_LEAD) | pl.col("t").list.last().is_in(WEB_MARK))
    for _ in range(2):
        df = df.with_columns(pl.when(had_web & pl.col("t").list.first().is_in(WEB_LEAD))
                             .then(pl.col("t").list.slice(1)).otherwise(pl.col("t")).alias("t"))
    for _ in range(3):
        df = df.with_columns(pl.when(had_web & (pl.col("t").list.len() > 1) & pl.col("t").list.last().is_in(WEB_TAIL))
                             .then(pl.col("t").list.slice(0, (pl.col("t").list.len().cast(pl.Int64) - 1).clip(lower_bound=0))).otherwise(pl.col("t")).alias("t"))
    for k in (4, 3, 2):  # spelled-out legal forms at the end: "s a s" -> "sas", "l l c" -> "llc"
        tail = pl.col("t").list.slice((pl.col("t").list.len().cast(pl.Int64) - k).clip(lower_bound=0), k)
        joined = tail.list.join("")
        single = tail.list.eval(pl.element().str.len_chars()).list.max() == 1
        df = df.with_columns(pl.when((pl.col("t").list.len() > k) & single & joined.is_in(LEGAL))
                             .then(pl.concat_list(pl.col("t").list.slice(0, (pl.col("t").list.len().cast(pl.Int64) - k).clip(lower_bound=0)), joined))
                             .otherwise(pl.col("t")).alias("t"))
    ms = (pl.col("t").list.get(0, null_on_oob=True) == "m") & (pl.col("t").list.get(1, null_on_oob=True) == "s")
    df = df.with_columns(pl.when(ms & (pl.col("t").list.len() > 2)).then(pl.col("t").list.slice(2))
                         .otherwise(pl.col("t")).alias("t"))
    for _ in range(2):
        df = df.with_columns(pl.when(pl.col("t").list.first().is_in(TITLES) & (pl.col("t").list.len() > 1))
                             .then(pl.col("t").list.slice(1)).otherwise(pl.col("t")).alias("t"))
    legal_words = (pl.concat_list(pl.col("name_legal_form").str.split(" "),
                                  pl.col("t").list.eval(pl.element().filter(pl.element().is_in(LEGAL))))
                   .list.eval(pl.element().replace(LEGAL_CANON).filter(pl.element() != ""))
                   .list.unique().list.sort().list.join(" "))
    core = pl.col("t").list.eval(pl.element().filter(~pl.element().is_in(LEGAL + STOP)))
    df = df.with_columns(legal_words.alias("legal"),
                         pl.when(core.list.len() > 0).then(core).otherwise(pl.col("t")).alias("c"))
    return df.select(
        "gid", "country", "indic", "legal",
        pl.col("c").list.join(" ").alias("core"),
        pl.col("c").list.join("").alias("ns"),
        pl.col("c").list.eval(pl.element().str.slice(0, 1)).list.join("").alias("ini"),
        pl.col("c").list.first().fill_null("").alias("first"),
        pl.col("c").list.last().fill_null("").alias("last"),
        pl.col("c").list.len().cast(pl.UInt8).alias("ntok"),
    ).with_columns(skeleton_expr(pl.col("ns")).alias("skel"))


def component_table(s: pl.DataFrame) -> pl.DataFrame:
    """Exploded address components: gid, country, pos, comp (cleaned, original script), comp_ascii,
    code (hand-typed state table on the ASCII form)."""
    c = (s.select("gid", "country", "addr_components",
                  pl.int_ranges(0, pl.col("addr_components").list.len()).alias("pos"))
         .explode(["addr_components", "pos"]).drop_nulls("addr_components")
         .with_columns(pl.col("addr_components").str.to_lowercase()
                       .str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " ").str.strip_chars().alias("comp"))
         .filter(pl.col("comp") != "").drop("addr_components"))
    rom = romanise(c["comp"])
    c = c.with_columns(pl.col("comp").replace(rom["k"], rom["v"]).alias("comp_ascii"))
    return c.join(state_table(), left_on=["country", "comp_ascii"], right_on=["country", "comp_key"], how="left")


def learn_native_states(s1c: pl.DataFrame, s23c: pl.DataFrame) -> pl.DataFrame:
    """FIT pairs only: native-script component -> S1 state code, kept when support and share are high."""
    labels = access.load_labels("FIT").select("s1_gid", "s23_gid")
    s1code = s1c.drop_nulls("code").sort("pos").group_by("gid").agg(pl.col("code").first())
    native = s23c.filter(pl.col("comp").str.contains(INDIC)).select("gid", "country", "comp")
    j = (labels.join(s1code.rename({"gid": "s1_gid"}), on="s1_gid")
         .join(native.rename({"gid": "s23_gid"}), on="s23_gid"))
    cnt = j.group_by("country", "comp", "code").len("n")
    tot = cnt.group_by("country", "comp").agg(pl.col("n").sum().alias("tot"))
    best = (cnt.sort("n", descending=True).group_by("country", "comp").head(1).join(tot, on=["country", "comp"])
            .filter((pl.col("n") >= NATIVE_MIN_SUPPORT) & (pl.col("n") / pl.col("tot") >= NATIVE_MIN_SHARE)))
    return best.select("country", "comp", pl.col("code").alias("native_code"), "n", "tot")


def address_fields(s: pl.DataFrame, comps: pl.DataFrame, native: pl.DataFrame) -> pl.DataFrame:
    c = comps.join(native.select("country", "comp", "native_code"), on=["country", "comp"], how="left")
    c = c.with_columns(pl.coalesce("code", "native_code").alias("code"))
    region = (c.drop_nulls("code").sort("pos").group_by("gid")
              .agg(pl.col("code").first().alias("region"), pl.col("pos").first().alias("rpos")))
    c = c.join(region, on="gid", how="left").filter(pl.col("rpos").is_null() | (pl.col("pos") != pl.col("rpos")))
    street_map = []
    for country in c["country"].unique().to_list():
        m = dict(STREET)
        m.update(STREET_BY_COUNTRY.get(country, {}))
        street_map.append(c.filter(pl.col("country") == country).with_columns(
            pl.col("comp_ascii").str.split(" ").list.eval(pl.element().replace(m)).list.join(" ").alias("canon")))
    c = pl.concat(street_map, how="vertical") if street_map else c.with_columns(pl.lit("").alias("canon"))
    per = (c.sort("pos").group_by("gid")
           .agg(pl.col("canon").str.join(" ").alias("addr"), pl.col("canon").first().alias("street")))
    last = comps.sort("pos").group_by("gid").agg(pl.col("comp_ascii").last().alias("lastc"))
    hn = pl.col("addr_house_number").fill_null("").str.to_lowercase().str.replace_all(r"[^0-9a-z]", "")
    base = s.select("gid", hn.str.strip_chars_start("0").alias("hn"),
                    pl.col("addr_zip").fill_null("").alias("zip"))
    base = base.with_columns(pl.col("hn").str.extract_all(r"\d").list.sort().list.join("").alias("hn_digits"))
    out = (base.join(per, on="gid", how="left").join(region.select("gid", "region"), on="gid", how="left")
           .join(last, on="gid", how="left"))
    return out.with_columns(pl.col("addr", "street", "region", "lastc").fill_null(""))


def load_source(dataset: str, src: int) -> pl.DataFrame:
    return C.load_silver_with_country(dataset, src).select(
        "gid", "country", "name_tokens", "name_legal_form", "addr_components", "addr_house_number", "addr_zip")


def add_shares() -> None:
    """Rewrites each records file with s1_share / s23_share, relative to its (dataset, country) pool."""
    for dataset in ("train", "test"):
        paths = {src: OUT / f"{dataset}_s{src}.parquet" for src in (1, 2, 3)}
        s1 = pl.read_parquet(paths[1], columns=["country", "core"])
        s23 = pl.concat([pl.read_parquet(paths[k], columns=["country", "core"]) for k in (2, 3)])
        c1 = s1.group_by("country", "core").len("n1").join(s1.group_by("country").len("N1"), on="country")
        c23 = s23.group_by("country", "core").len("n23").join(s23.group_by("country").len("N23"), on="country")
        del s1, s23
        for src, path in paths.items():
            r = pl.read_parquet(path).drop("s1_share", "s23_share", strict=False)
            n = r.height
            r = (r.join(c1, on=["country", "core"], how="left").join(c23, on=["country", "core"], how="left")
                 .with_columns((pl.col("n1").fill_null(0) / pl.col("N1").fill_null(1)).cast(pl.Float32).alias("s1_share"),
                               (pl.col("n23").fill_null(0) / pl.col("N23").fill_null(1)).cast(pl.Float32).alias("s23_share"))
                 .drop("n1", "N1", "n23", "N23"))
            if r.height != n:
                raise AssertionError(f"{path.name}: share join changed row count")
            r.write_parquet(path)


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"sources": {}}
    # pass 1: learn the native-script state map from FIT pairs (small frames only)
    s1code = (component_table(load_source("train", 1)).drop_nulls("code").sort("pos")
              .group_by("gid").agg(pl.col("code").first(), pl.col("country").first()))
    native_parts = []
    for src in (2, 3):
        c = component_table(load_source("train", src))
        native_parts.append(c.filter(pl.col("comp").str.contains(INDIC)).select("gid", "country", "comp", "pos", "code"))
        del c
    native = learn_native_states(s1code.with_columns(pl.lit(0).alias("pos")), pl.concat(native_parts))
    native.write_parquet(OUT / "native_state_map.parquet")
    report["native_state_map_rows"] = native.height
    print("native map", native.height, "rows", round(time.time() - t0, 1), "s", flush=True)
    del s1code, native_parts
    # pass 2: one source at a time
    for key in SOURCES:
        s = load_source(*key)
        r = name_fields(s).join(address_fields(s.select("gid", "addr_house_number", "addr_zip"),
                                               component_table(s), native), on="gid", how="left", validate="1:1")
        if r.height != s.height:
            raise AssertionError(f"{key}: row count changed")
        r.write_parquet(OUT / f"{key[0]}_s{key[1]}.parquet")
        report["sources"][f"{key[0]}_s{key[1]}"] = {
            "rows": r.height, "region_found": round(float((r["region"] != "").mean()), 4),
            "indic_names": int(r["indic"].sum()), "empty_core": int((r["core"] == "").sum())}
        print(key, report["sources"][f"{key[0]}_s{key[1]}"], round(time.time() - t0, 1), "s", flush=True)
        del s, r
    add_shares()
    report["runtime_s"] = round(time.time() - t0, 1)
    (OUT / "records_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
