from dataclasses import dataclass


@dataclass
class EntryMetadata:
    source_type: str
    source_id: str
    source_entry_id: str
