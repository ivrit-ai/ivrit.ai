from pydantic import BaseModel


class EntryMetadata(BaseModel):
    source_type: str
    source_id: str
    source_entry_id: str


class NormalizedEntryMetadata(EntryMetadata):
    """Common metadata fields for normalized entries across all sources."""

    quality_score: float | None = None
    per_segment_quality_scores: list[dict[str, float]] | None = None
    segments_count: int | None = None
    words_count: int | None = None
    avg_words_per_segment: float | None = None
    avg_segment_duration: float | None = None
    avg_words_per_minute: float | None = None
