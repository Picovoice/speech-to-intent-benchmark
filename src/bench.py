import logging as log
import os
from argparse import ArgumentParser
from sys import argv

from engine import *
from mix import *

log.basicConfig(format='', level=log.INFO)


def main():
    parser = ArgumentParser()
    parser.add_argument('--engine', choices=[x.value for x in Engines], required=True)
    parser.add_argument('--noise', required=True, choices=['cafe', 'kitchen'])
    parser.add_argument('--google_dialogflow_credential_path', required=(Engines.GOOGLE_DIALOGFLOW.value in argv))
    parser.add_argument('--google_dialogflow_project_id', required=(Engines.GOOGLE_DIALOGFLOW.value in argv))
    parser.add_argument('--ibm_watson_model_id', required=(Engines.IBM_WATSON.value in argv))
    parser.add_argument('--ibm_watson_custom_id', default=None)
    parser.add_argument('--ibm_watson_stt_apikey', required=(Engines.IBM_WATSON.value in argv))
    parser.add_argument('--ibm_watson_stt_url', required=(Engines.IBM_WATSON.value in argv))
    parser.add_argument('--ibm_watson_nlu_apikey', required=(Engines.IBM_WATSON.value in argv))
    parser.add_argument('--ibm_watson_nlu_url', required=(Engines.IBM_WATSON.value in argv))
    parser.add_argument('--microsoft_luis_luis_prediction_key', required=(Engines.MICROSOFT_LUIS.value in argv))
    parser.add_argument('--microsoft_luis_luis_endpoint_url', required=(Engines.MICROSOFT_LUIS.value in argv))
    parser.add_argument('--microsoft_luis_luis_app_id', required=(Engines.MICROSOFT_LUIS.value in argv))
    parser.add_argument('--microsoft_luis_speech_key', required=(Engines.MICROSOFT_LUIS.value in argv))
    parser.add_argument('--microsoft_luis_speech_endpoint_id', required=(Engines.MICROSOFT_LUIS.value in argv))
    parser.add_argument('--picovoice_rhino_access_key', required=(Engines.PICOVOICE_RHINO.value in argv))
    parser.add_argument('--snrs_db', nargs='+', default=list(range(24, 3, -3)))
    parser.add_argument('--sleep_sec', type=float, default=2.)
    args = parser.parse_args()

    engine = Engines(args.engine)

    engine_params = dict()
    for k, v in vars(args).items():
        if k.startswith(engine.value.lower()):
            engine_params[k.replace(f'{engine.value.lower()}_', '')] = v

    engine = Engine.create(x=engine, log=log, **engine_params)
    log.info(f'Initialized `{str(engine)}` engine')

    noise = args.noise
    snrs_db = sorted([float(x) for x in args.snrs_db])
    sleep_sec = args.sleep_sec

    run(noise=noise, snrs_ds=snrs_db)

    for snr_db in snrs_db:
        log.info(f'{noise} {snr_db} dB:')
        num_examples, num_errors = engine.process(
            folder=os.path.join(os.path.dirname(__file__), f'../data/speech/{noise}_{snr_db}db'),
            sleep_sec=sleep_sec)
        log.info(f"{num_examples} {num_errors} {num_errors / num_examples}")


if __name__ == "__main__":
    main()
