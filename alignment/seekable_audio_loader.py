import subprocess
import warnings
from stable_whisper.audio import AudioLoader, load_source


class SeekableAudioLoader(AudioLoader):
    def _audio_loading_process(self):
        """
        This is a copy of the original AudioLoader from stable_ts with an optimization
        to the loading process when using a file source + load_sections.
        When 'load_sections' is specified, and the first section has a start we can instruct
        ffmpeg to seek within the audio source saving the need for a "chunk by chunk" seek
        which takes time.
        Breakable alignment makes a lot of mid-file restarts and this makes a difference in the runtime.
        """
        if not isinstance(self.source, str) or not self._stream:
            return
        only_ffmpeg = False
        source = load_source(self.source, verbose=self.verbose, only_ffmpeg=only_ffmpeg, return_dict=True)
        if isinstance(source, dict):
            info = source
            source = info.pop('popen')
        else:
            info = None
        if info and info['duration']:
            self._duration_estimation = info['duration']
            if not self._stream and info['is_live']:
                warnings.warn('The audio appears to be a continuous stream but setting was set to `stream=False`.')

        if isinstance(source, subprocess.Popen):
            self._extra_process, stdin = source, source.stdout
        else:
            stdin = None
        try:
            seek_start_cmd_parts = []
            if self.load_sections:
                start_at = self.load_sections[0][0]
                if start_at:
                    self._prev_seek = int(round(start_at * self._sr)) # as if we seeked all the way
                    self._accum_samples = self._prev_seek # as if we seeked all the way
                    seek_start_cmd_parts = ['-ss', str(start_at)]
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI in PATH.
            cmd = [
                "ffmpeg",
                "-loglevel", "panic",
                "-nostdin",
                "-threads", "0",
                *seek_start_cmd_parts,
                "-i", self.source if stdin is None else "pipe:",
                "-f", "s16le",
                "-ac", "1",
                "-acodec", "pcm_s16le",
                "-ar", str(self._sr),
                "-"
            ]
            out = subprocess.Popen(cmd, stdin=stdin, stdout=subprocess.PIPE)

        except subprocess.SubprocessError as e:
            raise RuntimeError(f"Failed to load audio: {e}") from e

        return out