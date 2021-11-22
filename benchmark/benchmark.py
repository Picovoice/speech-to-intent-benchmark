import argparse
from sys import argv

from engine import *


def _path(x):
    return os.path.join(os.path.dirname(__file__), '../%s' % x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_type', choices=['AMAZON_LEX', 'GOOGLE_DIALOGFLOW', 'IBM_WATSON', 'MICROSOFT_LUIS',
                                                  'PICOVOICE_RHINO'],
                        required=True)
    parser.add_argument('--gcp_credential_path', required=('GOOGLE_DIALOGFLOW' in argv))
    parser.add_argument('--gcp_project_id', required=('GOOGLE_DIALOGFLOW' in argv))
    parser.add_argument('--ibm_credential_path', required=('IBM_WATSON' in argv))
    parser.add_argument('--ibm_model_id', required=('IBM_WATSON' in argv))
    parser.add_argument('--ibm_custom_id')
    parser.add_argument('--noise', required=True)
    parser.add_argument('--access-key')

    args = parser.parse_args()
    args_dict = vars(args)

    if args.access_key is not None:
        args_dict['ACCESS_KEY'] = args.access_key

    if args.gcp_credential_path is not None:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.gcp_credential_path

    if args.ibm_credential_path is not None:
        with open(args.ibm_credential_path) as f:
            for line in f:
                if len(line.strip()) > 0:
                    key, value = line.strip().split('=', maxsplit=1)
                    if key == 'SPEECH_TO_TEXT_APIKEY':
                        args_dict['stt_apikey'] = value
                    elif key == 'SPEECH_TO_TEXT_URL':
                        args_dict['stt_url'] = value
                    elif key == 'NATURAL_LANGUAGE_UNDERSTANDING_APIKEY':
                        args_dict['nlu_apikey'] = value
                    elif key == 'NATURAL_LANGUAGE_UNDERSTANDING_URL':
                        args_dict['nlu_url'] = value

    if args.engine_type == 'MICROSOFT_LUIS':
        with open(_path('data/luis/credentials.env')) as f:
            for line in f:
                if len(line.strip()) > 0:
                    key, value = line.strip().split('=', maxsplit=1)
                    args_dict[key] = value

    engine = NLUEngine.create(**args_dict)
    print('created %s engine' % str(engine))

    for snr_db in [24, 21, 18, 15, 12, 9, 6]:
        print('%s %d db:' % (args.noise, snr_db))
        if args.engine_type == 'PICOVOICE_RHINO':
            engine.process(_path('data/speech/%s_%ddb' % (args.noise, snr_db)), sleep_msec=0)
        else:
            engine.process(_path('data/speech/%s_%ddb' % (args.noise, snr_db)))
