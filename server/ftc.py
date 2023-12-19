from flask import Flask, redirect, render_template, request, url_for, session, jsonify
from flask_oauthlib.client import OAuth

import argparse
import base64
import dotenv
import json
import math
import os
import random
import threading

import utils
import mutagen.mp3
from sqlalchemy import func, Boolean, Float

dotenv.load_dotenv()

from db_models import Session, Transcript

app = Flask(__name__)
app.secret_key = os.environ['FLASK_APP_SECRET']
oauth = OAuth(app)

in_dev = 'FTC_STAGING_MODE' in os.environ

transcripts = None

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

transcribed_lock = threading.Lock()
transcribed_per_user = {}
transcribed_total = 0.0

def add_seconds_transcribed(user, seconds):
    global transcribed_total

    with transcribed_lock:
        if not user in transcribed_per_user:
            transcribed_duration = Session().query(func.sum(Transcript.data['payload']['duration'].cast(Float))).filter(Transcript.created_by == user).filter(Transcript.data['payload']['skipped'].cast(Boolean) == False).first()[0]

            if transcribed_duration == None:
                transcribed_duration = 0.0

            transcribed_per_user[user] = transcribed_duration

        transcribed_per_user[user] += seconds
        transcribed_total += seconds

        return transcribed_per_user[user]    

@app.route('/')
def index():
    if in_dev:
        session['user_email'] = os.environ['FTC_USER_EMAIL']

    if 'user_email' in session:
        return render_template('transcribe.html', user_name=session['user_email'])

    return redirect(url_for('login'))

@app.route('/login')
def login():
    return render_template('login.html')

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
        'user_seconds_transcribed' : add_seconds_transcribed(session['user_email'], 0.0),
        'total_seconds_transcribed' : transcribed_total
    })

@app.route('/api/submitResult', methods=['POST'])
def submit_content():
    data = request.get_json()
    source, episode, idx = data['uuid'].split('/')

    if not data['payload']['skipped']:
        add_seconds_transcribed(session['user_email'], data['payload']['duration'])

    with Session() as s:
        transcript_entry = Transcript(source=source, episode=episode, segment=idx, created_by=session['user_email'], data=data)
        s.add(transcript_entry)
        s.commit()
        s.close()

    return jsonify({'status': 'success', 'message': 'Data received successfully'})

if __name__ == '__main__':
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

    port = 5005 if in_dev else 4443
    app.run(host='0.0.0.0', port=port, ssl_context=('../secrets/certchain.pem', '../secrets/private.key'))

