"""gid <-> entity_id string mapping (e.g. gid 12345 <-> "S1-714132312"), needed only for the final
TSV output; internal pipeline code works in gid throughout."""
import polars as pl

from amlc.foundation import access


def load_gid_id_map(dataset: str, src: int) -> pl.DataFrame:
    """Returns (gid, entity_id)."""
    return access.load_bronze(dataset, src, columns=["gid", "entity_id"])
