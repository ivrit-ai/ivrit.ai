import warnings
from typing import List, Union, Optional

import numpy as np
import torch
from stable_whisper.non_whisper.alignment import (
    Aligner,
    AudioLoader,
    BasicWordTiming,
    WhisperResult,
    WordToken,
    format_timestamp,
    safe_print,
    tqdm,
)


class BreakableAligner(Aligner):
    """Extended Aligner that can break out of alignment when confidence is too low.

    This class extends the base Aligner to add functionality for monitoring alignment quality
    and breaking out of the alignment process when the quality falls below a threshold.
    It uses an exponential moving average of match probabilities to detect poor alignment.
    """

    def __init__(
        self,
        *args,
        match_probs_ema_alpha: float = 0.95,
        match_probs_ema_breakout_threshold: float = 0.5,
        match_probs_ema_warmup_ends_at: int = 180,
        **kwargs,
    ):
        """Initialize the BreakableAligner.

        Args:
            *args: Arguments to pass to the parent Aligner class.
            match_probs_ema_alpha: Alpha parameter for exponential moving average of match probabilities.
                Higher values give more weight to past observations (0.0-1.0).
            match_probs_ema_breakout_threshold: Threshold below which to break out of alignment.
                If the EMA of match probabilities falls below this value, alignment stops.
            match_probs_ema_warmup_ends_at: Time in seconds after which to start considering breakouts.
                Breakouts are not triggered before this time to allow for initial alignment.
            **kwargs: Additional keyword arguments to pass to the parent Aligner class.
        """
        self.match_probs_ema_alpha = match_probs_ema_alpha
        self.match_probs_ema_breakout_threshold = match_probs_ema_breakout_threshold
        self.match_probs_ema_warmup_ends_at = match_probs_ema_warmup_ends_at
        super().__init__(*args, **kwargs)

    def _update_pbar(
        self,
        tqdm_pbar: tqdm,
        last_ts: float,
        match_probs_ema: float = 0,
        finish: bool = False,
    ):
        """Update the progress bar with current alignment information.

        Args:
            tqdm_pbar: The progress bar to update.
            last_ts: The timestamp of the last processed segment.
            match_probs_ema: The exponential moving average of match probabilities.
            finish: Whether this is the final update.
        """
        if match_probs_ema is not None:
            tqdm_pbar.set_postfix({"probs": round(match_probs_ema, 2)})
        super()._update_pbar(tqdm_pbar, last_ts, finish)

    def align(
        self,
        audio: Union[str, np.ndarray, torch.Tensor, bytes, AudioLoader],
        text: Union[str, List[int], WhisperResult],
        **options,
    ) -> Union[WhisperResult, None]:
        self._reset()
        self._load_text(text)
        self._load_audio(audio)
        self._load_nonspeech_detector()
        for k in list(options.keys()):
            if hasattr(self, k):
                setattr(self, k, options.pop(k))
        self.options.update(options)

        with tqdm(
            total=self._initial_duration,
            unit="sec",
            disable=self.options.progress.verbose is not False,
            desc="Align",
        ) as tqdm_pbar:
            result: List[BasicWordTiming] = []
            last_ts = 0.0
            premature_stop = False
            match_probs_ema = None
            first_seen_ts = None

            while self._all_word_tokens:
                audio_segment, seek = self.audio_loader.next_valid_chunk(self._seek_sample, self.n_samples)
                self._seek_sample = seek
                self._time_offset = self._seek_sample / self.sample_rate
                if audio_segment is None:
                    break

                self._nonspeech_preds = self.nonspeech_predictor.predict(audio=audio_segment, offset=self._time_offset)

                audio_segment = self._skip_nonspeech(audio_segment)
                if audio_segment is None:
                    continue

                self._curr_words = self._compute_timestamps(audio_segment, *self._get_curr_words())
                self._seg_word_tokens = [WordToken(wts.word, wts.tokens) for wts in self._curr_words]

                last_ts = self._fallback(audio_segment.shape[-1])
                first_seen_ts = first_seen_ts if first_seen_ts is not None else last_ts

                if self._curr_words:
                    kept_words_probs = np.array([wt.probability for wt in self._curr_words])
                    mean_kept_words_probs = np.mean(kept_words_probs)
                    if match_probs_ema is None:
                        match_probs_ema = mean_kept_words_probs
                    match_probs_ema = (
                        mean_kept_words_probs * (1 - self.match_probs_ema_alpha)
                        + match_probs_ema * self.match_probs_ema_alpha
                    )

                self._update_pbar(tqdm_pbar, last_ts, match_probs_ema)

                result.extend(self._curr_words)

                if self.options.progress.verbose:
                    line = "\n".join(
                        f"[{format_timestamp(wts.start)}] -> " f'[{format_timestamp(wts.end)}] "{wts.word}"'
                        for wts in self._curr_words
                    )
                    safe_print(line)

                if self.failure_threshold is not None:
                    self.failure_count += sum(1 for wts in self._curr_words if wts.end - wts.start == 0)
                    if self.failure_count > self.max_fail:
                        premature_stop = True
                        tqdm_pbar.write("Breaking - too many 0-len segments")
                        break

                if (
                    self.match_probs_ema_warmup_ends_at < (last_ts - first_seen_ts)
                    and match_probs_ema is not None
                    and match_probs_ema < self.match_probs_ema_breakout_threshold
                ):
                    premature_stop = True
                    tqdm_pbar.write("Breaking - probs too low")
                    break

            self._update_pbar(tqdm_pbar, last_ts, match_probs_ema, self.failure_count <= self.max_fail)

        if not premature_stop and self._temp_data.word is not None:
            result.append(self._temp_data.word)
        if not result:
            warnings.warn("Failed to align text.", stacklevel=2)
        if self._all_word_tokens and not premature_stop:
            last_ts_str = format_timestamp(result[-1].end if result else 0)
            print(
                f"Failed to align the last {len(self._all_word_tokens)}/{self._total_words} words after "
                f"{last_ts_str}.",
            )

        if self._all_word_tokens and not self.remove_instant_words:
            final_total_duration = self.audio_loader.get_duration(3)
            result.extend(
                [
                    BasicWordTiming(
                        word=w.word,
                        start=final_total_duration,
                        end=final_total_duration,
                        tokens=w.tokens,
                        probability=0.0,
                    )
                    for w in self._all_word_tokens
                ]
            )

        self.audio_loader.terminate()
        self.nonspeech_predictor.finalize_timings()

        if not result:
            return

        final_result = [
            dict(
                word=w.word,
                start=w.start,
                end=w.end,
                tokens=w.tokens,
                probability=w.probability,
            )
            for w in result
        ]
        if len(self._split_indices_by_char):
            word_lens = np.cumsum([len(w.word) for w in result])
            split_indices = [np.flatnonzero(word_lens >= i)[0] + 1 for i in self._split_indices_by_char]
            final_result = WhisperResult(
                [final_result[i:j] for i, j in zip([0] + split_indices[:-1], split_indices) if i != j]
            )
        else:
            final_result = WhisperResult([final_result])

        self._suppress_silence(final_result)

        if not self.original_split:
            final_result.regroup(self.options.post.regroup)

        if fail_segs := len([None for s in final_result.segments if s.end - s.start <= 0]):
            print(
                f"{fail_segs}/{len(final_result.segments)} segments failed to align.",
            )

        return final_result


def breakable_align(
    model: "Whisper",
    audio: Union[str, np.ndarray, torch.Tensor, bytes, AudioLoader],
    text: Union[str, List[int], WhisperResult],
    language: str = None,
    *,
    tokenizer: "Tokenizer" = None,
    ignore_compatibility: bool = False,
    remove_instant_words: bool = False,
    token_step: int = 100,
    original_split: bool = False,
    word_dur_factor: Optional[float] = 2.0,
    max_word_dur: Optional[float] = 3.0,
    nonspeech_skip: Optional[float] = 5.0,
    fast_mode: bool = False,
    failure_threshold: Optional[float] = None,
    match_probs_ema_alpha: float = 0.95,
    match_probs_ema_breakout_threshold: float = 0.5,
    match_probs_ema_warmup_ends_at: int = 180,
    **options,
) -> Union[WhisperResult, None]:
    """Align text to audio with the ability to break out when confidence is too low.

    This function extends the standard alignment process by using the BreakableAligner,
    which can stop alignment when the quality falls below a threshold.

    Args:
        ... stable_ts params are documented in the base "align" function
        failure_threshold: Threshold for the ratio of failed words to total words.
        match_probs_ema_alpha: Alpha parameter for EMA of match probabilities.
        match_probs_ema_breakout_threshold: Threshold below which to break out of alignment.
        match_probs_ema_warmup_ends_at: Time in seconds after which to start considering breakouts.
        **options: Additional options to pass to the aligner.

    Returns:
        A WhisperResult containing the aligned text, or None if alignment failed.
    """
    from typing import TYPE_CHECKING, Optional

    is_faster_model = model.__module__.startswith("faster_whisper.")
    if not is_faster_model:
        from stable_whisper.whisper_compatibility import warn_compatibility_issues, whisper

        warn_compatibility_issues(whisper, ignore_compatibility)

    from stable_whisper.whisper_compatibility import N_SAMPLES, SAMPLE_RATE, TOKENS_PER_SECOND
    from stable_whisper.alignment import get_alignment_tokenizer, get_whisper_alignment_func, set_result_language
    from stable_whisper.options import AllOptions

    max_token_step = (model.max_length if is_faster_model else model.dims.n_text_ctx) - 6
    if token_step < 1:
        token_step = max_token_step
    elif token_step > max_token_step:
        raise ValueError(f"The max value for [token_step] is {max_token_step} but got {token_step}.")

    tokenizer, supported_languages = get_alignment_tokenizer(model, is_faster_model, text, language, tokenizer)

    options = AllOptions(options, vanilla_align=not is_faster_model)
    split_words_by_space = getattr(tokenizer, "language_code", tokenizer.language) not in {"zh", "ja", "th", "lo", "my"}
    model_type = "fw" if is_faster_model else None
    inference_func = get_whisper_alignment_func(model, tokenizer, model_type, options)

    aligner = BreakableAligner(
        inference_func=inference_func,
        decode=tokenizer.decode,
        encode=tokenizer.encode,
        split_words_by_space=split_words_by_space,
        sample_rate=SAMPLE_RATE,
        tokens_per_sec=TOKENS_PER_SECOND,
        max_segment_length=N_SAMPLES,
        remove_instant_words=remove_instant_words,
        token_step=token_step,
        original_split=original_split,
        word_dur_factor=word_dur_factor,
        max_word_dur=max_word_dur,
        nonspeech_skip=nonspeech_skip,
        fast_mode=fast_mode,
        failure_threshold=failure_threshold,
        match_probs_ema_alpha=match_probs_ema_alpha,
        match_probs_ema_breakout_threshold=match_probs_ema_breakout_threshold,
        match_probs_ema_warmup_ends_at=match_probs_ema_warmup_ends_at,
        all_options=options,
    )

    result = aligner.align(audio, text)
    set_result_language(result, tokenizer, language, supported_languages)

    return result
