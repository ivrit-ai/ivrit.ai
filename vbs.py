#!/usr/bin/env python3

import argparse
import json

from pydub import AudioSegment
from scipy.signal import find_peaks
import numpy as np

import matplotlib.pyplot as plt

verbose = False

def plot_rms_with_ranges(time_points, rms_values, ranges):
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, rms_values, label='RMS Value', color='blue')
    
    # Add range markers
    for start, end in ranges:
        plt.axvspan(start, end, color='green', alpha=0.3, label='Highlighted Range')
    
    plt.xlabel('Time (s)')
    plt.ylabel('RMS')
    plt.title('RMS Values Over Time with Highlighted Ranges')
    plt.grid(True)
    plt.show()

def load_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    return audio

def audio_to_rms(audio_segment, chunk_size=50):
    samples = np.array(audio_segment.get_array_of_samples())
    samples = samples.astype(np.float32) / 32768.0  # normalize

    rms = np.sqrt(np.convolve(samples ** 2, np.ones(chunk_size) / chunk_size, mode='valid'))

    return rms

def identify_contiguous_segments(rms, trigger_threshold, shutoff_threshold):
    segments = []

    i = 0
    while i < len(rms):
        if rms[i] >= trigger_threshold:
            segment = find_containing_segment(rms, i, shutoff_threshold)

            #if args.verbose:
            #    print(f'Identified containing segment, {segment[0]}-->{segment[1]}')

            segments.append(segment)
            i = segment[1] + 1
        else:
            i = i + 1

    return segments

def find_containing_segment(rms, base, shutoff_threshold, grace_range=50):
    start = base

    while start > 0:
        new_start = start

        lookback_start = max(0, start - grace_range)
        for j in range(lookback_start, start):
            if rms[j] >= shutoff_threshold:
                new_start = j
                break

        if new_start == start:
            break

        start = new_start 

    end = base
    while end < len(rms):
        new_end = end

        lookahead_end = min(len(rms) - 1, end + grace_range - 1) 
        for j in range(lookahead_end, end, -1):
            if rms[j] >= shutoff_threshold:
                new_end = j
                break

        if new_end == end:
            break

        end = new_end

    # Technically we're overflowing a bit, taking an the first element on each side that's below shutoff
    return [start, end]

def merge_nearby_segments(segments, min_silence_duration, min_segment_duration):
    merged_segments = []

    merged_seg = segments[0]
    for i in range(1, len(segments)):
       if (segments[i][0] - merged_seg[1]) < min_silence_duration:
           merged_seg = [merged_seg[0], segments[i][1]]
       else:
           merged_seg_duration = merged_seg[1] - merged_seg[0]
           if merged_seg_duration >= min_segment_duration:
               merged_segments.append(merged_seg)

           merged_seg = segments[i]

    merged_seg_duration = merged_seg[1] - merged_seg[0]
    if merged_seg_duration >= min_segment_duration:
        merged_segments.append(merged_seg)

    return merged_segments

def split_on_volume(audio, trigger_threshold=0.02, shutoff_threshold=0.005, min_silence_duration=0.5, chunk_size=50):
    rms = audio_to_rms(audio, chunk_size)


    segments = identify_contiguous_segments(rms, trigger_threshold, shutoff_threshold)

    frm = audio.frame_rate * audio.sample_width
    merged_segments = merge_nearby_segments(segments, int(min_silence_duration * frm), int(0.1 * frm))

    # Convert segment indices to real time 
    merged_segments = [[seg[0] / frm, seg[1] / frm] for seg in merged_segments]

    print(f'Number of segments: {len(merged_segments)}')
    for i in range(len(merged_segments)):
        delta = merged_segments[i][0] - merged_segments[i-1][1] if i > 0 else 0.0
        if verbose:
            print(f'{merged_segments[i][0]}-->{merged_segments[i][1]}, +{delta} from previous segment')

    if verbose:
        timestamps = [i / frm for i in range(len(rms))]
        #plot_rms_with_ranges(timestamps, rms, merged_segments)

    return merged_segments

def load_and_split(audio_filename):
    # Load audio file
    print('Loading audio...')
    audio = load_audio(audio_filename)

    # Split audio based on volume
    print('Splitting segments...')
    segments = split_on_volume(audio)

    return segments

def save_segments(audio_filename, segments, target_dir):
    audio = AudioSegment.from_file(audio_filename)

    for idx, s in enumerate(segments):
        start, end = s
        print(start, end, int(start*1000), int(end*1000))
        if idx > 0:
            delta = min(0.4, start - segments[idx-1][1])
            start -= delta / 2 

        if idx < len(segments) - 1:
            delta = min(0.4, segments[idx + 1][0] - end)
            end += delta / 2

        audio[int(start*1000) : int(end*1000)].export(f'{target_dir}/{idx}.mp3', format='mp3')

        if verbose:
            print(f'Done exporting segment #{idx}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split input file based on audio volume.')
    parser.add_argument('--input-file', type=str, required=True, help='Input file.')
    parser.add_argument('--verbose', type=bool, help='Verbose information.')

    args = parser.parse_args()
    verbose = args.verbose

    segments = load_and_split(args.input_file)

    json.dump(segments, open('segments.json', 'w'))

    save_segments(args.input_file, segments, '.')
