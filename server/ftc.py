from flask import Flask, Response, redirect, render_template, request, url_for, session, jsonify
from flask_oauthlib.client import OAuth

import argparse
import base64
import dotenv
import gzip
import io
import json
import math
import os
import random
import sortedcontainers
import threading

import utils
import mutagen.mp3
from box import Box
from sqlalchemy import func, Boolean, Float

dotenv.load_dotenv()

import db_models
from db_models import Session, Transcript

app = Flask(__name__)
app.secret_key = os.environ['FLASK_APP_SECRET']
oauth = OAuth(app)

in_dev = 'FTC_STAGING_MODE' in os.environ

transcripts = None
transcribed_total = 0.0

LEADERBOARD_TOP_N = 100
RETRANSCRIBE_RATE = 0.1

per_user_data = {}
sorted_per_user_data = sortedcontainers.SortedList([], key=lambda member: -member['duration'])

audio_dir = None
transcripts_dir = None

# Configure Google OAuth
google = oauth.remote_app(
    'google',
    consumer_key=os.environ['GOOGLE_CLIENT_ID'],
    consumer_secret=os.environ['GOOGLE_CLIENT_SECRET'],
    request_token_params={
        'scope': 'email'
    },
    base_url='https://www.googleapis.com/oauth2/v1/',
    request_token_url=None,
    access_token_method='POST',
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
)

def initialize_transcripts():
    global transcripts

    transcripts = []

    t_jsons = utils.find_files([transcripts_dir], '', ['.json'])

    if in_dev:
        t_jsons = t_jsons[0:1]

    for t_idx, t in enumerate(t_jsons):
        print(f'Processing ({t_idx}/{len(t_jsons)}) {t}...')
        j = json.load(open(t, 'r'))

        source = j['source']
        episode = j['episode']

        for idx, transcript in enumerate(j['transcripts']):
            texts = []
            max_logprob = -math.inf

            for seg in transcript['segments']:
                texts.append(seg['text'].strip())
                max_logprob = max(max_logprob, seg['avg_logprob'])

            text = ' '.join(texts).strip()

            transcripts.append((source, episode, idx, text, max_logprob)) 

    transcripts.sort(key=lambda e: e[4])

def initialize_per_user_data():
    for e in Session().query(func.count(Transcript.id), func.sum(Transcript.data['payload']['duration'].cast(Float)), Transcript.created_by).filter(Transcript.data['payload']['skipped'].cast(Boolean) == False).group_by(Transcript.created_by).all():
        user = e[2]

        data = { 'user' : user,
                 'segments' : e[0],
                 'duration' : e[1]
                }

        per_user_data[user] = data
        sorted_per_user_data.add(data)

transcribed_lock = threading.Lock()

def count_transcribed_segment(user, seconds):
    global transcribed_total

    with transcribed_lock:
        data = per_user_data[user]
        sorted_per_user_data.remove(data)

        data['segments'] += 1
        data['duration'] += seconds
        sorted_per_user_data.add(data)

        transcribed_total += seconds

@app.route('/')
def index():
    if in_dev:
        session['user_email'] = os.environ['FTC_USER_EMAIL']

    if not 'user_email' in session:
        return redirect(url_for('login'))

    # Ensure per_user_data is initialized
    user = session['user_email']
    if not user in per_user_data:
        data = { 'user' : user, 'segments' : 0, 'duration' : 0.0 }

        per_user_data[user] = data
        sorted_per_user_data.add(data)

    return render_template('transcribe.html',
                            user_name=session['user_email'],
                            google_analytics_tag=os.environ['GOOGLE_ANALYTICS_TAG'])

@app.route('/login')
def login():
    return render_template('login.html',
                           google_analytics_tag=os.environ['GOOGLE_ANALYTICS_TAG'])

@app.route('/logout')
def logout():
    session.pop('user_email')
    return redirect(url_for('index'))

@app.route('/authorize')
def authorize():
    return google.authorize(callback=url_for('authorized', _external=True))

@app.route('/login/authorized')
def authorized():
    resp = google.authorized_response()
    if resp is None or resp.get('access_token') is None:
        return 'Access denied: reason={0} error={1}'.format(
            request.args['error_reason'],
            request.args['error_description']
        )

    session['google_token'] = (resp['access_token'], '')
    session['user_email'] = google.get('userinfo').data["email"]
 
    session.pop('google_token')

    return redirect(url_for('index'))

@google.tokengetter
def get_google_oauth_token():
    return session.get('google_token')

@app.route('/api/getContent')
def get_content():
    # Once in a while, provide an already-transcribed segment
    if random.random() < RETRANSCRIBE_RATE:
        rtc = get_retranscribe_content()

        if rtc:
            return rtc

    global transcripts

    elem_index = random.choice(range(len(transcripts)))
    source, episode, idx, text, max_logprob = transcripts[elem_index]

    fn = f'{audio_dir}/{source}/{episode}/{idx}.mp3'

    return jsonify({
        'audioData': base64.b64encode(open(fn, 'rb').read()).decode('utf-8'),
        'text': text,
        'source': source,
        'episode': episode,
        'segment': idx,
        'uuid' : f'{source}/{episode}/{idx}',
        'duration' : mutagen.mp3.MP3(fn).info.length,
        'complexity' : 10 - round(10 * elem_index / len(transcripts), 1),
        'max_logprob' : max_logprob,
        'attributes' : {}
    })

def get_retranscribe_content():

    elem = None

    for i in range(3):
        max_idx = db_models.Session().query(func.max(db_models.Transcript.id)).first()[0]
        idx = random.randrange(0, max_idx + 1)
        elem = db_models.Session().query(db_models.Transcript).filter(db_models.Transcript.id == idx).first()

        if elem:
            break

    if not elem:
        return None

    source = elem.source
    episode = elem.episode
    idx = elem.segment
    text = elem.data['payload']['text']
    max_logprob = elem.data['payload']['max_logprob']

    fn = f'{audio_dir}/{source}/{episode}/{idx}.mp3'

    return jsonify({
        'audioData': base64.b64encode(open(fn, 'rb').read()).decode('utf-8'),
        'text': text,
        'source': source,
        'episode': episode,
        'segment': idx,
        'uuid' : f'{source}/{episode}/{idx}',
        'duration' : mutagen.mp3.MP3(fn).info.length,
        # Hard coded complexity for now; needs a list search in transcripts for accuracy
        'complexity' : 5,
        'max_logprob' : max_logprob,
        'attributes' : { 'retranscribe' : True }
    })


@app.route('/api/getStatistics')
def get_statistics():
    user = session['user_email']
    stats = Box()

    with transcribed_lock:
        stats.user = session['user_email']
        stats.user_seconds_transcribed = per_user_data[user]['duration']
        stats.total_seconds_transcribed = transcribed_total

        set_ranking_info(user, stats)

        return jsonify(stats)

def set_ranking_info(user, stats):
    idx = sorted_per_user_data.bisect_key_left(-stats.user_seconds_transcribed)

    stats.num_transcribers = len(sorted_per_user_data)
    stats.percentile = 100 * (idx / stats.num_transcribers)

    if idx == 0:
        stats.rank = 1
        stats.next_duration = 0.0
        stats.next_rank = 0
        stats.text = 'God-like!'
    else:
        prev_idx = sorted_per_user_data.bisect_key_left(-sorted_per_user_data[idx - 1]['duration'])

        stats.rank = idx + 1
        stats.next_duration = sorted_per_user_data[prev_idx]['duration'] - stats.user_seconds_transcribed
        stats.next_rank = prev_idx + 1
        stats.text = 'Go go go!'

@app.route('/api/submitResult', methods=['POST'])
def submit_content():
    data = request.get_json()
    user_agent = request.headers.get('User-Agent')
    referer = request.headers.get('Referer')

    source, episode, idx = data['uuid'].split('/')

    data['payload']['stats']['user_agent'] = user_agent
    data['payload']['stats']['referer'] = referer

    if not data['payload']['skipped']:
        count_transcribed_segment(session['user_email'], seconds=data['payload']['duration'])

    with Session() as s:
        transcript_entry = Transcript(source=source, episode=episode, segment=idx, created_by=session['user_email'], data=data)
        s.add(transcript_entry)
        s.commit()
        s.close()

    return jsonify({'status': 'success', 'message': 'Data received successfully'})

# Ability to dump entire database directly from the server.
# Not enabled yet, until there's access control in place.
#@app.route('/api/fetchDatabase')
def fetch_database():
    if not 'user_email' in session:
        return redirect(url_for('login'))

    jdata = db_models.table_to_json_str(db_models.Transcript)
        
    gzip_buffer = io.BytesIO()
    with gzip.GzipFile(mode='wb', fileobj=gzip_buffer) as gzip_file:
        gzip_file.write(jdata.encode('utf-8'))  # Write the bytes of the JSON string
    
    # Get the byte value of the compressed data
    gzip_data = gzip_buffer.getvalue()
    
    # Send the gzipped response
    return Response(
        gzip_data,
        mimetype='application/gzip',
        headers={"Content-Disposition": "attachment;filename=data.json.gz"}
    )

def obfuscate_email(email):
    parts = email.split("@")
    local = parts[0]
    domain = parts[1]
    obfuscated_local = local[0] + "*" * (len(local) - 2) + local[-1]
    return f"{obfuscated_local}@{domain}"

def format_duration(seconds):
    """Converts duration from seconds to HH:MM:SS format."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

@app.route('/leaderboard')
def leaderboard():
    # Sort users based on duration
    sorted_users = sorted(per_user_data.items(), key=lambda x: x[1]['duration'], reverse=True)
    
    # Obfuscate email, format duration, add ranking, and prepare display data
    display_data = [{
        "rank": idx + 1,
        "segments": data["segments"],
        "duration": format_duration(data["duration"]),
        "email": obfuscate_email(email)
    } for idx, (email, data) in enumerate(sorted_users)]

    # Only show the top N
    display_data = display_data[:LEADERBOARD_TOP_N]

    # Render the template with the users data
    return render_template('leaderboard.html', users=display_data)


parser = argparse.ArgumentParser(description='Launch transcription server.')

# Add the arguments
parser.add_argument('--audio-dir', type=str, required=True,
                    help='Root directory for audio files.')
parser.add_argument('--transcripts-dir', type=str, required=True,
                    help='Root directory for transcripts.')

# Parse the arguments
args = parser.parse_args()
audio_dir = args.audio_dir
transcripts_dir = args.transcripts_dir 

initialize_transcripts()
print(f'Done loading {len(transcripts)} transcripts.')

transcribed_total = Session().query(func.sum(Transcript.data['payload']['duration'].cast(Float))).filter(Transcript.data['payload']['skipped'].cast(Boolean) == False).first()[0]

initialize_per_user_data()

if __name__ == '__main__':
    port = 5005 if in_dev else 4443
    app.run(host='0.0.0.0', port=port, ssl_context=('secrets/certchain.pem', 'secrets/private.key'))
