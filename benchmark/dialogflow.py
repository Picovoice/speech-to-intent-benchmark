import json
import os
import sys
import time

import dialogflow_v2 as dialogflow


def _path(x):
    return os.path.join(os.path.dirname(__file__), '../%s' % x)


def process_file(path, project_id):
    cache_path = path.replace('.wav', '.dialogflow')

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    session_client = dialogflow.SessionsClient()

    session_id = os.path.basename(path)[0]

    session = session_client.session_path(project_id, session_id)

    with open(path, 'rb') as audio_file:
        input_audio = audio_file.read()

    audio_config = dialogflow.types.InputAudioConfig(
        audio_encoding=dialogflow.enums.AudioEncoding.AUDIO_ENCODING_LINEAR_16,
        language_code='en',
        sample_rate_hertz=16000)

    query_input = dialogflow.types.QueryInput(audio_config=audio_config)

    response = session_client.detect_intent(session=session, query_input=query_input, input_audio=input_audio)

    result = dict(intent=response.query_result.intent.display_name, slots=dict())

    for k in ['coffeeDrink', 'size', 'roast', 'numberOfShots', 'sugarAmount', 'milkAmount']:
        if response.query_result.parameters.fields.get(k) is not None:
            v = response.query_result.parameters.fields.get(k).string_value
            if v != '':
                if k == 'size':
                    if '8' in v or 'eight' in v:
                        v = 'eight ounce'
                    elif '12' in v or 'twelve' in v:
                        v = 'twelve ounce'
                    elif '16' in v or 'sixteen' in v:
                        v = 'sixteen ounce'
                    elif '20' in v or 'twenty' in v:
                        v = 'twenty ounce'
                result['slots'][k] = v

    with open(cache_path, 'w') as f:
        json.dump(result, f)

    return result


def process(folder, project_id, sleep_msec=2, retry_limit=32):
    with open(_path('data/label/label.json')) as f:
        labels = json.load(f)

    num_examples = 0
    num_errors = 0
    for x in os.listdir(folder):
        if x.endswith('.wav'):
            num_examples += 1

            if x not in labels:
                raise ValueError("the label for '%s' is missing" % x)
            label = labels[x]

            time.sleep(sleep_msec)

            attempts = 0
            intent = None
            while attempts < retry_limit:
                try:
                    intent = process_file(os.path.join(folder, x), project_id)
                    break
                except Exception as ex:
                    print(ex)
                    attempts += 1

            if attempts == retry_limit:
                raise RuntimeError()

            if intent is None:
                num_errors += 1
                continue

            if label["intent"] != intent["intent"]:
                num_errors += 1
                continue
            for slot in label["slots"].keys():
                if slot not in intent["slots"]:
                    num_errors += 1
                    break
                if intent["slots"][slot].strip() != label["slots"][slot].strip():
                    num_errors += 1
                    break

    print('num examples: %d' % num_examples)
    print('num errors: %d' % num_errors)
    print('accuracy: %f' % (float(num_examples - num_errors) / num_examples))


if __name__ == '__main__':
    credential_path = sys.argv[1]
    gcp_project_id = sys.argv[2]
    noise = sys.argv[3]
    snrs = [24, 21, 18, 15, 12, 9, 6]

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

    for snr_db in snrs:
        print('%s %d db:' % (noise, snr_db))
        process(_path('data/speech/%s_%ddb' % (noise, snr_db)), gcp_project_id)
