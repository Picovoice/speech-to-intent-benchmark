import json
import os
import sys
import time
import uuid

import boto3


def _path(x):
    return os.path.join(os.path.dirname(__file__), '../%s' % x)


def process_file(path):
    cache_path = path.replace('.wav', '.amazonlex')

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    client = boto3.client('lex-runtime')

    with open(path, 'rb') as input_audio:
        response = client.post_content(
            botName='barista',
            botAlias='$LATEST',
            userId=str(uuid.uuid4()),
            contentType='audio/l16; rate=16000; channels=1',
            accept='text/plain; charset=utf-8',
            inputStream=input_audio
        )

    result = dict(intent=response['intentName'],
                  slots=response['slots'],
                  inputTranscript=response['inputTranscript'])

    for k in ['coffeeDrink', 'size', 'roast', 'numberOfShots', 'sugarAmount', 'milkAmount']:
        if k in result['slots'] and result['slots'][k] is not None:
            v = result['slots'][k]
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
        json.dump(result, f, indent=2)

    return result


def process(folder, sleep_msec=2, retry_limit=32):
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
            result = None
            while attempts < retry_limit:
                try:
                    result = process_file(os.path.join(folder, x))
                    break
                except Exception as ex:
                    print(ex)
                    attempts += 1

            if attempts == retry_limit:
                raise RuntimeError()

            if result is None:
                num_errors += 1
                continue

            if label['intent'] != result['intent']:
                num_errors += 1
                continue

            for slot in result['slots'].keys():
                if result['slots'][slot] is None:
                    if slot in label['slots'].keys():
                        num_errors += 1
                        break
                elif slot in label['slots'].keys():
                    if result['slots'][slot].strip() != label['slots'][slot].strip():
                        num_errors += 1
                        break
                elif slot not in label['slots'].keys():
                    num_errors += 1
                    break

    print('num examples: %d' % num_examples)
    print('num errors: %d' % num_errors)
    print('accuracy: %f' % (float(num_examples - num_errors) / num_examples))


if __name__ == '__main__':
    noise = sys.argv[1]
    snrs = [24, 21, 18, 15, 12, 9, 6]

    for snr_db in snrs:
        print('%s %d db:' % (noise, snr_db))
        process(_path('data/speech/%s_%ddb' % (noise, snr_db)))
