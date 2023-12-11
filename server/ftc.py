from flask import Flask, redirect, render_template, request, url_for, session, jsonify
from flask_oauthlib.client import OAuth

import base64
import dotenv
import json
import math
import os
import random

import utils

dotenv.load_dotenv()

import db_models

app = Flask(__name__)
app.secret_key = os.environ['FLASK_APP_SECRET']
oauth = OAuth(app)

transcripts = None

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

    t_jsons = utils.find_files(['/media/yair/big/ivrit.ai/transcripts-new'], '', ['.json'])[0:100]
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

@app.route('/')
def index():
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

    elem_index = 1000 + random.choice(range(len(transcripts[1000:])))
    source, episode, idx, text, max_logprob = transcripts[elem_index]

    fn = f'/media/yair/big/ivrit.ai/splits-new/{source}/{episode}/{idx}.mp3'

    return jsonify({
        'audioData': base64.b64encode(open(fn, 'rb').read()).decode('utf-8'),
        'text': text,
        'source': source,
        'episode': episode,
        'uuid' : f'{source}/{episode}/{idx}',
        'complexity' : 10 - round(10 * elem_index / len(transcripts), 1)
    })

@app.route('/api/submitResult', methods=['POST'])
def submit_content():
    data = request.get_json()
    source, episode, idx = data['uuid'].split('/')

    print(data)

    with db_models.Session() as s:
        transcript_entry = db_models.Transcript(source=source, episode=episode, segment=idx, created_by=session['user_email'], data=data)
        s.add(transcript_entry)
        s.commit()
        s.close()

    return jsonify({'status': 'success', 'message': 'Data received successfully'})

if __name__ == '__main__':
    initialize_transcripts()
    print(f'Done loading {len(transcripts)} transcripts.')
    app.run(host='0.0.0.0', port=4443, ssl_context=('../secrets/certchain.pem', '../secrets/private.key'))

