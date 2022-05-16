import json
import os
import time
import uuid
from enum import Enum
from logging import Logger
from typing import *

import azure.cognitiveservices.speech as speechsdk
import boto3
import dialogflow_v2 as dialogflow
import pvrhino
import requests
import soundfile
import urllib3
from azure.cognitiveservices.language.luis.runtime import LUISRuntimeClient
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import NaturalLanguageUnderstandingV1, SpeechToTextV1
from ibm_watson.natural_language_understanding_v1 import EntitiesOptions, Features
from msrest.authentication import CognitiveServicesCredentials

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _path(x):
    return os.path.join(os.path.dirname(__file__), '../%s' % x)


class Engines(Enum):
    AMAZON_LEX = 'AMAZON_LEX'
    GOOGLE_DIALOGFLOW = 'GOOGLE_DIALOGFLOW'
    IBM_WATSON = 'IBM_WATSON'
    MICROSOFT_LUIS = 'MICROSOFT_LUIS'
    PICOVOICE_RHINO = 'PICOVOICE_RHINO'


class Engine(object):
    def __init__(self, log: Optional[Logger] = None) -> None:
        self._log = log

    def process_file(self, path: str) -> Optional[Dict[str, str]]:
        raise NotImplementedError()

    def process(self, folder: str, sleep_msec: float = 2., retry_limit: int = 32) -> Tuple[int, int]:
        with open(os.path.join(os.path.dirname(__file__), f'../data/label/label.json')) as f:
            labels = json.load(f)

        num_examples = 0
        num_errors = 0
        for x in os.listdir(folder):
            if x.endswith('.wav'):
                num_examples += 1

                label = labels[x]

                time.sleep(sleep_msec)

                retry_count = 0
                inference = None
                while retry_count < retry_limit:
                    try:
                        inference = self.process_file(os.path.join(folder, x))
                        break
                    except Exception as e:
                        if self._log is not None:
                            self._log.warning(e)
                        retry_count += 1
                if retry_count == retry_limit:
                    raise RuntimeError()

                if inference is None:
                    num_errors += 1
                elif label["intent"] != inference["intent"]:
                    num_errors += 1
                else:
                    for slot in label["slots"].keys():
                        if slot not in inference["slots"]:
                            num_errors += 1
                            break
                        if inference["slots"][slot].strip() != label["slots"][slot].strip():
                            num_errors += 1
                            break

        return num_examples, num_errors

    def __str__(self) -> str:
        raise NotImplementedError()

    @classmethod
    def create(cls, x: Engines, **kwargs: Any) -> 'Engine':
        if x is Engines.AMAZON_LEX:
            return AmazonLex()
        elif x is Engines.GOOGLE_DIALOGFLOW:
            return GoogleDialogflow(credential_path=kwargs['gcp_credential_path'], project_id=kwargs['gcp_project_id'])
        elif x is Engines.IBM_WATSON:
            return IBMWatson(
                kwargs['ibm_model_id'],
                kwargs['ibm_custom_id'],
                kwargs['stt_apikey'],
                kwargs['stt_url'],
                kwargs['nlu_apikey'],
                kwargs['nlu_url'])
        elif x is Engines.MICROSOFT_LUIS:
            return MicrosoftLUIS(
                kwargs['LUIS_PREDICTION_KEY'],
                kwargs['LUIS_ENDPOINT_URL'],
                kwargs['LUIS_APP_ID'],
                kwargs['SPEECH_KEY'],
                kwargs['SPEECH_ENDPOINT_ID'])
        elif x is Engines.PICOVOICE_RHINO:
            return PicovoiceRhino(access_key=kwargs['access_key'])
        else:
            raise ValueError(f"Cannot create {cls.__name__} of type `{x.value}`")


class AmazonLex(Engine):
    def __init__(self, log: Optional[Logger] = None) -> None:
        super(AmazonLex, self).__init__(log=log)
        self._client = boto3.client('lex-runtime')

    def process_file(self, path: str) -> Optional[Dict[str, str]]:
        cache_path = path.replace('.wav', '.lex')

        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)

        with open(path, 'rb') as f:
            response = self._client.post_content(
                botName='barista',
                botAlias='$LATEST',
                userId=str(uuid.uuid4()),
                contentType='audio/l16; rate=16000; channels=1',
                accept='text/plain; charset=utf-8',
                inputStream=f
            )

        res = {
            "intent": response['intentName'],
            "slots": {k: v for k, v in response['slots'].items() if v is not None},
            "transcript": response['inputTranscript']
        }

        with open(cache_path, 'w') as f:
            json.dump(res, f, indent=2)

        return res

    def __str__(self) -> str:
        return Engines.AMAZON_LEX.value


class GoogleDialogflow(Engine):
    def __init__(self, credential_path: str, project_id: str, log: Optional[Logger] = None) -> None:
        super(GoogleDialogflow, self).__init__(log=log)

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

        self._project_id = project_id

    def process_file(self, path):
        cache_path = path.replace('.wav', '.dialogflow')

        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)

        session_client = dialogflow.SessionsClient()

        session_id = os.path.basename(path)[0]

        session = session_client.session_path(self._project_id, session_id)

        with open(path, 'rb') as f:
            input_audio = f.read()

        audio_config = dialogflow.types.InputAudioConfig(
            audio_encoding=dialogflow.enums.AudioEncoding.AUDIO_ENCODING_LINEAR_16,
            language_code='en',
            sample_rate_hertz=16000)

        query_input = dialogflow.types.QueryInput(audio_config=audio_config)

        response = session_client.detect_intent(session=session, query_input=query_input, input_audio=input_audio)

        result = dict(intent=response.query_result.intent.display_name, slots=dict())

        for k, v in response.query_result.parameters.fields.items():
            v = v.string_value
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

    def __str__(self):
        return Engines.GOOGLE_DIALOGFLOW.value


class IBMWatson(Engine):
    def __init__(
            self,
            model_id,
            custom_id,
            stt_apikey,
            stt_url,
            nlu_apikey,
            nlu_url,
            log: Optional[Logger] = None) -> None:
        super(IBMWatson, self).__init__(log=log)

        self._model_id = model_id
        self._username = "apikey"
        self._stt_apikey = stt_apikey
        self._stt_url = stt_url
        self._nlu_apikey = nlu_apikey
        self._nlu_url = nlu_url
        self._headers = {'Content-Type': "application/json"}
        if custom_id is None:
            self._custom_id = self._create_language_model()
            self._train_language_model()
        else:
            self._custom_id = custom_id

    def _create_language_model(self):
        data = {"name": "barista_1", "base_model_name": "en-US_BroadbandModel",
                "description": "STT custom model for coffee maker context"}
        uri = self._stt_url + "/v1/customizations"
        response = requests.post(uri, auth=(self._username, self._stt_apikey), verify=False,
                                 headers=self._headers, data=json.dumps(data).encode('utf-8'))

        if response.status_code != 201:
            print(response.text)
            raise RuntimeError("Failed to create model")

        custom_id = response.json()['customization_id']
        print("Model customization id: ", custom_id)
        return custom_id

    def _add_corpus(self):
        corpus_name = "corpus1"
        corpus_path = _path('data/watson/corpus.txt')

        uri = self._stt_url + "/v1/customizations/" + self._custom_id + "/corpora/" + corpus_name
        with open(corpus_path, 'rb') as f:
            response = requests.post(uri, auth=(self._username, self._stt_apikey), verify=False,
                                     headers=self._headers, data=f)

        if response.status_code != 201:
            print(response.text)
            raise RuntimeError("Failed to add corpus file")

        print("Added corpus file")
        return uri

    def _get_corpus_status(self, uri):
        response = requests.get(uri, auth=(self._username, self._stt_apikey), verify=False, headers=self._headers)
        response_json = response.json()
        status = response_json['status']
        time_to_run = 0
        while status != 'analyzed' and time_to_run < 10000:
            time.sleep(10)
            response = requests.get(uri, auth=(self._username, self._stt_apikey), verify=False, headers=self._headers)
            response_json = response.json()
            status = response_json['status']
            time_to_run += 10

        if status != 'analyzed':
            raise RuntimeError()

        print("Corpus analysis complete")

    def _train(self):
        uri = self._stt_url + "/v1/customizations/" + self._custom_id + "/train"
        response = requests.post(uri, auth=(self._username, self._stt_apikey),
                                 verify=False, data=json.dumps({}).encode('utf-8'))

        if response.status_code != 200:
            raise RuntimeError("Failed to start training custom model")

        print("Started training custom model")

    def _get_training_status(self):
        uri = self._stt_url + "/v1/customizations/" + self._custom_id
        response = requests.get(uri, auth=(self._username, self._stt_apikey), verify=False, headers=self._headers)
        status = response.json()['status']
        time_to_run = 0
        while status != 'available' and time_to_run < 10000:
            time.sleep(10)
            response = requests.get(uri, auth=(self._username, self._stt_apikey), verify=False, headers=self._headers)
            status = response.json()['status']
            time_to_run += 10

        if status != 'available':
            raise RuntimeError()

        print("Training custom model complete")

    def _train_language_model(self):
        corpus_uri = self._add_corpus()
        self._get_corpus_status(corpus_uri)
        self._train()
        self._get_training_status()

    def process_file(self, path):
        cache_path = path.replace('.wav', '.watson')

        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)

        stt_service = SpeechToTextV1(authenticator=IAMAuthenticator(self._stt_apikey))
        stt_service.set_service_url(self._stt_url)

        with open(path, 'rb') as audio_file:
            stt_response = stt_service.recognize(
                audio=audio_file,
                content_type='audio/wav',
                language_customization_id=self._custom_id
            ).get_result()['results']

        if stt_response:
            transcript = stt_response[0]['alternatives'][0]['transcript'].lower()
        else:
            return None

        nlu_service = NaturalLanguageUnderstandingV1(authenticator=IAMAuthenticator(self._nlu_apikey),
                                                     version='2018-03-16')
        nlu_service.set_service_url(self._nlu_url)

        response = nlu_service.analyze(
            features=Features(entities=EntitiesOptions(model=self._model_id)),
            text=transcript,
            language='en'
        ).get_result()['entities']

        intent = None
        slots = dict()
        for e in response:
            if e['type'] == 'orderDrink':
                intent = 'orderDrink'
            else:
                slots[e['type']] = e['text']

        result = dict(intent=intent, slots=slots, transcript=transcript)

        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    def __str__(self):
        return 'IBM Watson'


class MicrosoftLUIS(Engine):
    def __init__(
            self,
            prediction_key,
            endpoint_url,
            app_id,
            speech_key,
            speech_endpoint_id,
            log: Optional[Logger] = None) -> None:
        super(MicrosoftLUIS, self).__init__(log=log)

        self._initial_silence_timeout_ms = 15000
        self._slot_name = 'staging'
        self._region = 'westus'
        self._prediction_key = prediction_key
        self._endpoint_url = endpoint_url
        self._app_id = app_id
        self._speech_key = speech_key
        self._speech_endpoint_id = speech_endpoint_id

    def process_file(self, path):
        cache_path = path.replace('.wav', '.luis')

        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)

        template = "wss://{}.stt.speech.microsoft.com/speech/recognition" \
                   "/conversation/cognitiveservices/v1?initialSilenceTimeoutMs={:d}"
        speech_config = speechsdk.SpeechConfig(subscription=self._speech_key,
                                               endpoint=template.format(self._region,
                                                                        self._initial_silence_timeout_ms))
        source_language_config = speechsdk.languageconfig.SourceLanguageConfig("en-US", self._speech_endpoint_id)
        audio_config = speechsdk.audio.AudioConfig(filename=path)

        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, source_language_config=source_language_config, audio_config=audio_config)
        speech_response = speech_recognizer.recognize_once()

        if speech_response is None:
            return None

        if speech_response.reason == speechsdk.ResultReason.RecognizedSpeech:
            transcript = speech_response.text.lower()
        elif speech_response.reason == speechsdk.ResultReason.NoMatch:
            raise Exception(speech_response.no_match_details)
        elif speech_response.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = speech_response.cancellation_details
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))
            raise Exception(cancellation_details.reason)
        else:
            return None

        if not transcript:
            return None
        request = {"query": transcript}

        client_runtime = LUISRuntimeClient(self._endpoint_url, CognitiveServicesCredentials(self._prediction_key))
        nlu_response = client_runtime.prediction.get_slot_prediction(app_id=self._app_id, slot_name=self._slot_name,
                                                                     prediction_request=request)

        if nlu_response is None:
            return None

        intent = nlu_response.prediction.top_intent
        slots = dict()
        for k, v in nlu_response.prediction.entities.items():
            slots[k] = [e[0].strip() for e in v]

        result = dict(intent=intent, slots=slots, transcript=transcript)

        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    def process(self, folder, sleep_msec=2, retry_limit=32):
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
                        result = self.process_file(os.path.join(folder, x))
                        break
                    except Exception as ex:
                        print(ex)
                        attempts += 1

                if attempts == retry_limit:
                    raise RuntimeError()

                if result is None:
                    num_errors += 1
                    continue

                if label["intent"] != result["intent"]:
                    num_errors += 1
                    continue

                for slot in label["slots"].keys():
                    if slot not in result["slots"]:
                        num_errors += 1
                        break
                    if label["slots"][slot].strip() not in result["slots"][slot]:
                        num_errors += 1
                        break

        print('num examples: %d' % num_examples)
        print('num errors: %d' % num_errors)
        print('accuracy: %f' % (float(num_examples - num_errors) / num_examples))

    def __str__(self):
        return 'Microsoft LUIS'


class PicovoiceRhino(Engine):
    def __init__(self, access_key: str, sensitivity: float = 0.75, log: Optional[Logger] = None) -> None:
        super(PicovoiceRhino, self).__init__(log=log)
        self._o = pvrhino.create(
            access_key=access_key,
            context_path=os.path.join(os.path.dirname(__file__), '../data/rhino/coffee_maker_linux.rhn'),
            sensitivity=sensitivity)

    def process_file(self, path: str) -> Optional[Dict[str, str]]:
        pcm, sample_rate = soundfile.read(path, dtype='int16')
        assert pcm.ndim == 1
        assert sample_rate == self._o.sample_rate

        is_finalized = False
        start_index = 0
        while start_index < (len(pcm) - self._o.frame_length) and not is_finalized:
            end_index = start_index + self._o.frame_length
            is_finalized = self._o.process(pcm[start_index: end_index])
            start_index = end_index

        if not is_finalized:
            result = None
        else:
            inference = self._o.get_inference()
            if inference.is_understood:
                result = dict(intent=inference.intent, slots=inference.slots)
            else:
                result = None

        return result

    def __str__(self) -> str:
        return Engines.PICOVOICE_RHINO.value


__all__ = [
    'Engines',
    'Engine',
]
