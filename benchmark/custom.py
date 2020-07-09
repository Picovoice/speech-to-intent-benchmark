import json
import os
import time

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
headers = {'Content-Type': "application/json"}


def _path(x):
    return os.path.join(os.path.dirname(__file__), '../%s' % x)


def get_credentials():
    file = os.getenv('IBM_CREDENTIALS_FILE')
    env_vars = {}
    with open(file) as f:
        for line in f:
            if not line.strip():
                continue
            key, value = line.strip().split('=', 1)
            env_vars[key] = value
    password = env_vars['SPEECH_TO_TEXT_APIKEY']
    url = env_vars['SPEECH_TO_TEXT_URL']
    return password, url


def create_custom_model(username, password, url):
    data = {"name": "barista_1", "base_model_name": "en-US_BroadbandModel",
            "description": "STT custom model for coffee maker context"}
    uri = url + "/v1/customizations"
    response = requests.post(uri, auth=(username, password), verify=False,
                             headers=headers, data=json.dumps(data).encode('utf-8'))

    if response.status_code != 201:
        print(response.text)
        raise RuntimeError("Failed to create model")

    custom_id = response.json()['customization_id']
    print("Model customization_id: ", custom_id)
    return custom_id


def add_corpus(username, password, url, custom_id, corpus_path):
    corpus_name = "corpus1"

    uri = url + "/v1/customizations/" + custom_id + "/corpora/" + corpus_name
    with open(corpus_path, 'rb') as f:
        response = requests.post(uri, auth=(username, password), verify=False, headers=headers, data=f)

    if response.status_code != 201:
        print(response.text)
        raise RuntimeError("Failed to add corpus file")

    print("Added corpus file")
    return uri


def get_corpus_status(username, password, uri):
    response = requests.get(uri, auth=(username, password), verify=False, headers=headers)
    response_json = response.json()
    status = response_json['status']
    time_to_run = 0
    while status != 'analyzed' and time_to_run < 10000:
        time.sleep(10)
        response = requests.get(uri, auth=(username, password), verify=False, headers=headers)
        response_json = response.json()
        status = response_json['status']
        time_to_run += 10

    if status != 'analyzed':
        raise RuntimeError()

    print("Corpus analysis complete")


def train(username, password, url, custom_id):
    uri = url + "/v1/customizations/" + custom_id + "/train"
    response = requests.post(uri, auth=(username, password), verify=False, data=json.dumps({}).encode('utf-8'))

    if response.status_code != 200:
        raise RuntimeError("Failed to start training custom model")


def get_training_status(username, password, url, custom_id):
    uri = url + "/v1/customizations/" + custom_id
    response = requests.get(uri, auth=(username, password), verify=False, headers=headers)
    status = response.json()['status']
    time_to_run = 0
    while status != 'available' and time_to_run < 10000:
        time.sleep(10)
        response = requests.get(uri, auth=(username, password), verify=False, headers=headers)
        status = response.json()['status']
        time_to_run += 10

    if status != 'available':
        raise RuntimeError()

    print("Training custom model complete")


def create_language_model():
    username = "apikey"
    password, url = get_credentials()
    custom_id = create_custom_model(username, password, url)
    corpus_uri = add_corpus(username, password, url, custom_id, _path('data/watson/corpus.txt'))
    get_corpus_status(username, password, corpus_uri)
    train(username, password, url, custom_id)
    get_training_status(username, password, url, custom_id)
    return custom_id
