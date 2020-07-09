import argparse
import os
from sys import argv

from engine import *


def _path(x):
    return os.path.join(os.path.dirname(__file__), '../%s' % x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine_type', choices=['AMAZON_LEX', 'GOOGLE_DIALOGFLOW', 'PICOVOICE_RHINO', 'IBM_WATSON'],
                        required=True)
    parser.add_argument('--gcp_credential_path', required=('GOOGLE_DIALOGFLOW' in argv))
    parser.add_argument('--gcp_project_id', required=('GOOGLE_DIALOGFLOW' in argv))
    parser.add_argument('--ibm_credential_path', required=('IBM_WATSON' in argv))
    parser.add_argument('--ibm_model_id', required=('IBM_WATSON' in argv))
    parser.add_argument('--ibm_custom_id')
    parser.add_argument('--noise', required=True)

    args = parser.parse_args()

    if args.gcp_credential_path is not None:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.gcp_credential_path

    if args.ibm_credential_path is not None:
        os.environ['IBM_CREDENTIALS_FILE'] = args.ibm_credential_path

    engine = NLUEngine.create(NLUEngines[args.engine_type], args.gcp_project_id, args.ibm_model_id, args.ibm_custom_id)
    print('created %s engine' % str(engine))

    for snr_db in [24, 21, 18, 15, 12, 9, 6]:
        print('%s %d db:' % (args.noise, snr_db))
        if args.engine_type == 'PICOVOICE_RHINO':
            engine.process(_path('data/speech/%s_%ddb' % (args.noise, snr_db)), sleep_msec=0)
        else:
            engine.process(_path('data/speech/%s_%ddb' % (args.noise, snr_db)))
