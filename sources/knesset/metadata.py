from typing import Optional

from sources.common.metadata import NormalizedEntryMetadata

source_type = "knesset"
plenum_source_id = "plenum"
committee_source_id = "committee"


class PlenumMetadata(NormalizedEntryMetadata):
    """Metadata for a Knesset plenum."""

    plenum_id: str
    duration: Optional[float] = None
    title: Optional[str] = None
    plenum_date: Optional[str] = None
