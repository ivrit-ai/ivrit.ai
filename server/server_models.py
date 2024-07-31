from __future__ import annotations
from typing import Iterable, List, Optional
from enum import Enum

from faster_whisper.transcribe import Segment, TranscriptionInfo, Word
from pydantic import BaseModel, Field


def segments_text(segments: list[Segment]) -> str:
    return "".join(segment.text for segment in segments).strip()


def words_from_segments(segments: list[Segment]) -> list[Word]:
    words = []
    for segment in segments:
        if segment.words is None:
            continue
        words.extend(segment.words)
    return words


class ResponseFormat(Enum):
    TEXT = "text"
    JSON = "json"
    VERBOSE_JSON = "verbose_json"

    def __str__(self):
        return self.value


class TimestampGranularity(str, Enum):
    WORD = "word"
    SEGMENT = "segment"


class WordObject(BaseModel):
    start: float
    end: float
    word: str
    probability: float

    @classmethod
    def from_word(cls, word: Word) -> WordObject:
        return cls(
            start=word.start,
            end=word.end,
            word=word.word,
            probability=word.probability,
        )


class SegmentObject(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float

    @classmethod
    def from_segment(cls, segment: Segment) -> SegmentObject:
        return cls(
            id=segment.id,
            seek=segment.seek,
            start=segment.start,
            end=segment.end,
            text=segment.text,
            tokens=segment.tokens,
            temperature=segment.temperature,
            avg_logprob=segment.avg_logprob,
            compression_ratio=segment.compression_ratio,
            no_speech_prob=segment.no_speech_prob,
        )


class TranscriptionJsonResponse(BaseModel):
    text: str

    @classmethod
    def from_segments(cls, segments: list[Segment]) -> TranscriptionJsonResponse:
        return cls(text=segments_text(segments))


class TranscriptionVerboseJsonResponse(BaseModel):
    task: str = "transcribe"
    language: str
    duration: float
    text: str
    words: list[WordObject]
    segments: list[SegmentObject]

    @classmethod
    def from_segment(cls, segment: Segment, transcription_info: TranscriptionInfo) -> TranscriptionVerboseJsonResponse:
        return cls(
            language=transcription_info.language,
            duration=segment.end - segment.start,
            text=segment.text,
            words=([WordObject.from_word(word) for word in segment.words] if isinstance(segment.words, list) else []),
            segments=[SegmentObject.from_segment(segment)],
        )

    @classmethod
    def from_segments(
        cls, segments: list[Segment], transcription_info: TranscriptionInfo
    ) -> TranscriptionVerboseJsonResponse:
        return cls(
            language=transcription_info.language,
            duration=transcription_info.duration,
            text=segments_text(segments),
            segments=[SegmentObject.from_segment(segment) for segment in segments],
            words=[WordObject.from_word(word) for word in words_from_segments(segments)],
        )


class CreateTranscriptionRequest(BaseModel):
    file: bytes = Field(
        ...,
        description="The audio file object (not file name) to transcribe, in one of the supported ffmpeg source format.",
    )
    model: str = Field(
        ...,
        description="ID of the model to use. Ignored for this server.",
    )
    language: Optional[str] = Field(
        None,
        description="The language of the input audio. Supplying the input language in [ISO-639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) format will improve accuracy and latency.",
    )
    prompt: Optional[str] = Field(
        None,
        description="An optional text to guide the model's style or continue a previous audio segment. The prompt should match the audio language.",
    )
    response_format: ResponseFormat = Field(
        ResponseFormat.JSON,
        description="The format of the transcript output, in one of these options: `json`, `text`, `verbose_json`",
    )
    temperature: Optional[float] = Field(
        default=0.0,
        description="The sampling temperature, between 0 and 1. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. If set to 0, the model will use log probability to automatically increase the temperature until certain thresholds are hit.",
        ge=0.0,
        le=1.0,
    )
    timestamp_granularities: Optional[List[TimestampGranularity]] = Field(
        [TimestampGranularity.SEGMENT],
        description="The timestamp granularities to populate for this transcription. `response_format` must be set `verbose_json` to use timestamp granularities. Either or both of these options are supported: `word`, or `segment`. Note: There is no additional latency for segment timestamps, but generating word timestamps incurs additional latency.",
    )


def segments_to_response(
    segments: Iterable[Segment],
    transcription_info: TranscriptionInfo,
    response_format: ResponseFormat,
) -> str | TranscriptionJsonResponse | TranscriptionVerboseJsonResponse:
    segments = list(segments)
    if response_format == ResponseFormat.TEXT:
        return segments_text(segments)
    elif response_format == ResponseFormat.JSON:
        return TranscriptionJsonResponse.from_segments(segments)
    elif response_format == ResponseFormat.VERBOSE_JSON:
        return TranscriptionVerboseJsonResponse.from_segments(segments, transcription_info)
