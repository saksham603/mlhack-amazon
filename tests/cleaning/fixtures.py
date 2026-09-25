"""Real bronze rows, identified by gid (CG-7). gids were looked up on 2026-09-26 from the strings
quoted in the Stage 1 prompt §1 and from the C1-M measurement examples; tests load the row live
and never retype its text."""
from functools import cache

import polars as pl

from amlc.foundation.access import load_bronze

# §1 quoted examples: (dataset, src, gid) -> exact expected C1 output (address, parts removed)
QUOTED_NULL = {
    ("train", 2, 2206869): ("067 production ct, independence, ky", 1),
    ("train", 3, 7241478): ("mulberry dr, buckeye, arizona", 1),
    ("test", 2, 14259633): ("70 barbaras way, newberg citty, or", 1),
    ("test", 3, 19146894): ("door no 543 139g/a, mettupalayam, coimbatore, tn", 1),
    ("train", 1, 283440): ("56-2-6/1, patamata, benz circle, vijayawada (urban), krishna, andhra pradesh", 1),
    ("test", 1, 12571669): ("41 quai de malakoff, nantes, pays de la loire", 1),
}
# every S1 row with a Stage 0 null part (MANIFEST_v2: exactly 8 train, 29 test)
S1_NULL_GIDS = {
    "train": [130884, 176779, 283440, 414890, 588967, 790344, 1175952, 1711578],
    "test": [12528605, 12552034, 12571669, 12575377, 12878270, 12944934, 12997068, 13005209, 13024748,
             13059392, 13084850, 13205875, 13290932, 13316546, 13364280, 13388488, 13528568, 13700790,
             13788569, 13807019, 14006439, 14010097, 14019501, 14040227, 14046709, 14068197, 14085491,
             14121300, 14207274],
}
# D-C1a case variants
VARIANT_NULL = {
    ("train", 1, 2005): ("rajasthan, vaishnav vihar nangal, jaipur, plot no- 153", 1),
    ("test", 3, 19186645): ("bardhaman, durgapur municipal corporation, wb", 2),
    ("train", 2, 3903400): ("499, second floor sector 52, gurgaon, haryana", 1),
}
# D-C1b garbled text, D-C1c ligature: (dataset, src, gid, column) -> exact expected clean text
GARBLED = {
    ("train", 1, 862, "business_address"):
        "a-212, malhotra complex, gali no. – 01, vikas marg, shakarpur, laxmi nagar, east delhi, delhi",
    ("train", 2, 2209781, "business_address"): "a/195., people’s co-operative colony, kankarbagh, patna, bihar",
    ("test", 1, 12532072, "business_address"):
        "#302 challa’s janakiram residency, 1-65/14/6/14, guttala begumpet, serilingamaplly, madha, pur, "
        "hyderabad, rangareddy, telangana",
    ("test", 1, 12947843, "business_name"): "d’aide & freres sarl",
    ("train", 1, 134301, "business_address"): "flat no 14284, prestige lakeside habitat, bangalore north, bangalore, karnataka",
    ("test", 1, 12566702, "business_address"): "41 avenue des oeillets, dunkerque, hauts-de-france",
    ("test", 1, 12566702, "business_name"): "lycee bde",
    # Derived from the spec (CG-17): part 'Ï¿½JAIPURÏ¿½-Ï¿½' -> each marker becomes a space (D-C1b)
    # -> ' JAIPUR - ' -> casefold -> collapse -> 'jaipur - ,' -> step 6 removes the space before ','
    ("test", 2, 14291910, "business_address"):
        "h.no a-49a, yojna shwroop vihar, gram todiramjanipura, jagatupura, jaipur -, jaipur, राजस्थान",
}
# D-C1e ASCII controls: (dataset, src, gid, column) -> exact expected clean text, derived from the spec
CONTROL = {
    # 'Medchal\x1aMalkajgiri' -> SUB becomes a space, so the two words stay separate
    ("train", 1, 674887, "business_name"): "medchal malkajgiri housekeeping private limited",
    # 'Shopper\x1aS Stop' -> 'shopper s stop' (SUB stood for a lost apostrophe)
    ("train", 1, 76749, "business_address"):
        "second floor vasant towers, bearing municipal no. 1-11-251/1b, behind shopper s stop, begu, mpet, "
        "hyderabad, telangana",
    # 'A/2 \x1aIcc' -> 'a/2 icc' (double space collapsed); 'Trade Tower\x1a,' -> 'trade tower ,' -> step 6 -> 'trade tower,'
    ("train", 1, 97016, "business_address"):
        "unit no.201, 2nd floor, wing b, f/p no.403 a/2 icc, trade tower, senapati bapat road, shivaji nagar, "
        "pune, maharashtra",
}
# D-C1f parts emptied by cleaning: (dataset, src, gid) -> (expected addr_clean, parts emptied), from the spec
EMPTIED = {
    # parts '\x1aRaman & Raman Shopping Complex\x1a' (kept, SUBs -> spaces) and '\x1a' (emptied -> dropped)
    ("test", 1, 12576029): ("raman & raman shopping complex, 48, thiruvidaimarudur road, kumbakonam, thanjavur, tamil nadu", 1),
    # 'Â\x80\x93' decodes to '–' (D-C1b); the lone '\x1a' part is emptied and dropped
    ("train", 2, 2286248): ("delhi, unit no.30, floor 3, worldmark – 3, asset – 7, aerocity, n.h – 8, new delhi, new delhi", 1),
}
# Indic names: C1 must leave them equal to NFC(raw) minus Cf characters
INDIC_NAMES = [("train", 2, 2206842), ("train", 2, 2207259), ("train", 2, 2207204), ("train", 2, 2207537)]


@cache
def _table(dataset: str, src: int) -> pl.DataFrame:
    return load_bronze(dataset, src, columns=["gid", "country", "business_name", "business_address"])


def row(dataset: str, src: int, gid: int) -> dict:
    r = _table(dataset, src).filter(pl.col("gid") == gid)
    assert r.height == 1, f"gid {gid} not found exactly once in {dataset} S{src}"
    return r.row(0, named=True)
