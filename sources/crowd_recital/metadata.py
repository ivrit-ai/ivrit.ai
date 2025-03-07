from typing import Optional

from sources.common.metadata import NormalizedEntryMetadata

source_type = "crowd_recital"
source_id = "recital"


class SessionMetadata(NormalizedEntryMetadata):
    """Metadata for a crowd recital session."""
    session_id: str
    session_duration: float
    user_id: str
    document_language: str
    document_title: str
    document_source_type: str
    year_of_birth: Optional[int] = None
    biological_sex: Optional[str] = None
