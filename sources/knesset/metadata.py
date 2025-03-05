from dataclasses import dataclass

from sources.metadata import EntryMetadata

source_type = "knesset"
plenum_source_id = "plenum"
committee_source_id = "committee"


@dataclass
class PlenumMetadata(EntryMetadata):
    plenum_id: str
    duration: float | None = None
    title: str | None = None
    quality_score: float | None = None
    per_segment_quality_scores: list[dict[str, float]] | None = None
    segments_count: int | None = None
    words_count: int | None = None
    avg_words_per_segment: float | None = None
    avg_segment_duration: float | None = None
    avg_words_per_minute: float | None = None
