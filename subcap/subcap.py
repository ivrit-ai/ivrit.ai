import re
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']

def timecode_to_seconds(timecode):
    hours, minutes, seconds, frames = map(int, timecode.split(':'))
    return hours * 3600 + minutes * 60 + seconds + frames / 25  # assuming 25 frames per second

def parse_subcap(file_path):
    encoding = detect_encoding(file_path)
    
    content = open(file_path, 'r', encoding=encoding, errors='ignore').read()

def parse_subcap(file_path):
    encoding = detect_encoding(file_path)
    
    with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
        file_content = file.read()
        
    subtitles = []
    subtitle_block = False
    lines = file_content.splitlines()
    
    for i, line in enumerate(lines):
        if "<begin subtitles>" in line:
            subtitle_block = True
            continue
        elif "<end subtitles>" in line:
            subtitle_block = False
            continue
        
        if subtitle_block:
            match = re.match(r"(\d{2}:\d{2}:\d{2}:\d{2})\s+(\d{2}:\d{2}:\d{2}:\d{2})", line)
            if match:
                start_time, end_time = match.groups()
                start_time_seconds = timecode_to_seconds(start_time) - 3600
                end_time_seconds = timecode_to_seconds(end_time) - 3600
                st_t1 = lines[i + 1].strip() if i + 1 < len(lines) else ""
                st_t2 = lines[i + 2].strip() if i + 2 < len(lines) else ""
                subtitle_text = st_t1 if st_t2 == "" else f"{st_t1} {st_t2}"
                subtitles.append({
                    "start_time": start_time_seconds,
                    "end_time": end_time_seconds,
                    "text": subtitle_text
                })
    
    return subtitles
