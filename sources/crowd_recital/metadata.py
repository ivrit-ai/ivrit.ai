from dataclasses import dataclass

from sources.metadata import EntryMetadata

source_type = "crowd_recital"
source_id = "recital"


@dataclass
class SessionMetadata(EntryMetadata):
    session_id: str
    session_duration: float
    user_id: str
    document_language: str
    document_title: str
    document_source_type: str
    year_of_birth: int | None
    biological_sex: str | None
    quality_score: float | None = None
    per_segment_quality_scores: list[dict[str, float]] | None = None
    segments_count: int | None = None
    words_count: int | None = None
    avg_words_per_segment: float | None = None
    avg_segment_duration: float | None = None
    avg_words_per_minute: float | None = None
