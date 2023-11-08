import logging

logging.basicConfig(format='%(asctime)s %(message)s')
logging.warning('Launching tc.py...')

import streamlit as st
import extra_streamlit_components as stx
import st_keyup
import diff_match_patch

from google_auth_oauthlib.flow import Flow
from cryptography.fernet import Fernet

import json
import requests

salt = open('salt.txt').read() 
key = Fernet(salt)

# Define the redirect URL after successful authentication
redirect_uri = 'https://serve.ivrit.ai'

st.title('ivrit.ai transcription')

params = st.experimental_get_query_params()

code = params['code'][0] if 'code' in params else None
state = params['state'][0] if 'state' in params else None

def get_user_email(redirect_uri, state, code):
    # Load credentials from JSON file
    j = json.load(open('client_secret.json'))

    client_id = j['web']['client_id']
    client_secret = j['web']['client_secret']


    token_url = 'https://oauth2.googleapis.com/token'
    token_data = {
        'code' : code,
        'client_id' : client_id,
        'client_secret' : client_secret,
        'redirect_uri' : redirect_uri,
        'grant_type' : 'authorization_code',
        'state' : state
    }

    token_response = requests.post(token_url, data=token_data)

    token_response_data = token_response.json()
    if not 'access_token' in token_response_data:
        return None

    access_token = token_response_data['access_token']

    userinfo_url = 'https://www.googleapis.com/oauth2/v3/userinfo'
    userinfo_response = requests.get(userinfo_url, headers={'Authorization': f'Bearer {access_token}'})
    userinfo_response_data = userinfo_response.json()

    email = userinfo_response_data['email']

    return email

def redirect(url):
    nav_script = """
        <meta http-equiv="refresh" content="0; url='%s'">
    """ % (url)
    st.write(nav_script, unsafe_allow_html=True)

#hashed_user_cookie = cm.get('hashed_user')
#if hashed_user_cookie:
#    st.session_state.hashed_user = hashed_user_cookie

hashed_user = st.session_state.hashed_user if 'hashed_user' in st.session_state else None

if code and state and not hashed_user:
    user = get_user_email(redirect_uri, state, code)
    if user == None:
        redirect(redirect_uri)
        st.stop()

    logging.warning('Storing hashed_user in cookie manager...') 
    st.session_state.hashed_user = key.encrypt(user.encode())

    st.experimental_set_query_params()
    st.rerun()

if hashed_user == None:
    if st.button('Log in with Google'):
        # Define the scopes required for your app
        scopes = ['email']

        # Create the OAuth flow instance
        flow = Flow.from_client_secrets_file(
            'client_secret.json',
            scopes=scopes,
            redirect_uri=redirect_uri
        )

        authorization_url, state = flow.authorization_url(prompt='consent')

        # Store the state in session state
        st.session_state.state = state

        # Redirect the user to the authorization URL
        redirect(authorization_url)

    st.stop()

user = key.decrypt(hashed_user).decode()
st.write(f'Logged in as {user}')

if not 'inited' in st.session_state:
    logging.warning('Initializing state...')
    st.session_state.orig_text = 'Space - it says - is big. Really big.'
    st.session_state.user_input = st.session_state['orig_text'] 

    mp3_file = "x.mp3"
    st.session_state.audio_bytes = open(mp3_file, 'rb').read() 

st.session_state.inited = True

# Serve the MP3 file
logging.warning('Serving audio...')
st.audio(st.session_state.audio_bytes, format='audio/mpeg')

# Calculate the diff
dmp = diff_match_patch.diff_match_patch()
diff = dmp.diff_main(st.session_state.orig_text, st.session_state.user_input)
dmp.diff_cleanupSemantic(diff)

# Format the diff for HTML
html = []
for (op, data) in diff:
    text = data.replace("\n", "<br>")
    if op == dmp.DIFF_INSERT:
        html.append(f"<span style='background: #e6ffe6;'>{text}</span>")
    elif op == dmp.DIFF_DELETE:
        html.append(f"<span style='background: #ffe6e6; text-decoration: line-through;'>{text}</span>")
    elif op == dmp.DIFF_EQUAL:
        html.append(text)
html = "".join(html)

# Add a read-only text box with the diff
st.markdown(html, unsafe_allow_html=True)

# Add a read-write text box
st_keyup.st_keyup("Enter some text", st.session_state.user_input, key='user_input')

# Add a submit button
if st.button('Submit'):
    st.write(f"You submitted: {st.session_state.user_input}")

