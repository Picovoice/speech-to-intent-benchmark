import logging as log
import os
from argparse import ArgumentParser
from sys import argv

from engine import *
from mixer import *

log.basicConfig(level=log.INFO)


def main():
    parser = ArgumentParser()
    parser.add_argument('--engine', choices=[x.value for x in Engines], required=True)
    parser.add_argument('--noise', required=True, choices=['cafe', 'kitchen'])
    parser.add_argument('--google_dialogflow_credential_path', required=(Engines.GOOGLE_DIALOGFLOW.value in argv))
    parser.add_argument('--google_dialogflow_project_id', required=(Engines.GOOGLE_DIALOGFLOW.value in argv))
    parser.add_argument('--ibm_watson_model_id', required=(Engines.IBM_WATSON.value in argv))
    parser.add_argument('--ibm_watson_custom_id', required=(Engines.IBM_WATSON.value in argv))
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
    parser.add_argument('--snrs_db', choices='+', default=[24, 21, 18, 15, 12, 9, 6])
    args = parser.parse_args()

    args.engine = Engines(args.engine)

    kwargs = dict()
    for k, v in vars(args).items():
        if k.startswith(args.engine.value.lower()):
            kwargs[k.replace(f'{args.engine.value.lower()}_', '')] = v

    engine = Engine.create(x=args.engine, log=log, **kwargs)
    log.info(f'created {args.engine.value} engine')

    run(noise=args.noise, snrs_ds=args.snrs_db)

    for snr_db in args.snrs_db:
        log.info(f'{args.noise} {snr_db} dB:')
        num_examples, num_errors = engine.process(
            os.path.join(os.path.dirname(__file__), f'../data/speech/{args.noise}_{snr_db}db'))
        log.info(f"{num_examples} {num_errors} {num_errors / num_examples}")


if __name__ == "__main__":
    main()
