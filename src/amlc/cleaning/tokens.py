"""Data tables for C3/C4 (Stage 1 prompt rev 5, §3). Open sets keyed by the country value seen in
the data (CG-4): a country with no entry falls through unchanged, never silently dropped.
"""

# C2: Unicode block ranges for per-token script classification (§3 C2).
SCRIPT_RANGES = [
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("odia", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
]
LATIN_ALPHA = [(0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0x1E00, 0x1EFF)]

# C3: dotted-abbreviation collapse, longer/overlapping patterns first (§3 C3).
# The trailing boundary is a negative lookahead, not \b: \.?\b backtracks off the final dot
# whenever the string ends right there (both sides of a bare "." are non-word), which would
# silently leave the dot unconsumed. (?![A-Za-z]) has no such backtracking interaction.
DOTTED_ABBREV = [
    (r"\bP\.?L\.?L\.?C\.?(?![A-Za-z])", "pllc"),
    (r"\bL\.?L\.?C\.?(?![A-Za-z])", "llc"),
    (r"\bP\.?C\.?(?![A-Za-z])", "pc"),
    (r"\bPvt\.(?![A-Za-z])", "pvt"),
    (r"\bLtd\.(?![A-Za-z])", "ltd"),
    (r"\bInc\.(?![A-Za-z])", "inc"),
    (r"\bCorp\.(?![A-Za-z])", "corp"),
    (r"\bCo\.(?![A-Za-z])", "co"),
]

HONORIFICS = ["mr", "mrs", "ms", "dr", "m/s", "shri", "smt"]

# C3: legal-form tables, country-aware (§3 C3). Canonical form is the dict value; both spellings
# map to it so India's pvt/private and ltd/limited never split a business into two blocking keys.
LEGAL_FORMS = {
    "US": {"inc": "inc", "llc": "llc", "pllc": "pllc", "pc": "pc", "corp": "corp", "co": "co",
           "ltd": "ltd", "esq": "esq", "dds": "dds", "md": "md"},
    "India": {"pvt": "pvt", "private": "pvt", "ltd": "ltd", "limited": "ltd", "llp": "llp", "opc": "opc"},
    "France": {"sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa",
               "snc": "snc", "sci": "sci", "selarl": "selarl"},
}
# India multi-word forms are matched before single tokens (longest match first): "pvt ltd" / "private limited".
LEGAL_FORM_PHRASES = {
    "India": [(("pvt", "ltd"), "pvt ltd"), (("private", "limited"), "pvt ltd")],
}

# C4: street-word canonicalization, country-aware (§3 C4). France: st/ste -> saint/sainte, never street.
STREET_WORDS = {
    "US": {"st": "street", "rd": "road", "dr": "drive", "ave": "avenue", "cir": "circle",
           "ln": "lane", "ct": "court", "blvd": "boulevard"},
    "India": {"rd": "road", "nr": "near", "opp": "opposite", "flr": "floor"},
    "France": {"r": "rue", "bd": "boulevard", "av": "avenue", "pl": "place", "imp": "impasse",
               "ch": "chemin", "st": "saint", "ste": "sainte"},
}

HOUSE_NUMBER_STRIP_WORDS = ["h.no", "h no", "door no", "no", "#"]
