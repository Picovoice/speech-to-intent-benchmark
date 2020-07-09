import json
import os
import sys
import time
import uuid
from enum import Enum

import boto3
import dialogflow_v2 as dialogflow
from ibm_watson import SpeechToTextV1
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions
import soundfile

from custom import create_language_model


def _path(x):
    return os.path.join(os.path.dirname(__file__), '../%s' % x)


class NLUEngines(Enum):
    AMAZON_LEX = 'AMAZON_LEX'
    GOOGLE_DIALOGFLOW = 'GOOGLE_DIALOGFLOW'
    PICOVOICE_RHINO = 'PICOVOICE_RHINO'
    IBM_WATSON = 'IBM_WATSON'


class NLUEngine(object):
    def process_file(self, path):
        raise NotImplementedError()

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
                    if result["slots"][slot].strip() != label["slots"][slot].strip():
                        num_errors += 1
                        break

        print('num examples: %d' % num_examples)
        print('num errors: %d' % num_errors)
        print('accuracy: %f' % (float(num_examples - num_errors) / num_examples))

    def __str__(self):
        raise NotImplementedError()

    @classmethod
    def create(cls, engine_type, gcp_project_id, ibm_model_id, ibm_custom_id):
        if engine_type is NLUEngines.AMAZON_LEX:
            return AmazonLex()
        elif engine_type is NLUEngines.GOOGLE_DIALOGFLOW:
            return GoogleDialogflow(gcp_project_id)
        elif engine_type is NLUEngines.IBM_WATSON:
            return IBMWatson(ibm_model_id, ibm_custom_id)
        elif engine_type is NLUEngines.PICOVOICE_RHINO:
            return PicovoiceRhino()
        else:
            raise ValueError("cannot create %s of type '%s'" % (cls.__name__, engine_type))


class AmazonLex(NLUEngine):
    def __init__(self):
        self._client = boto3.client('lex-runtime')

    def process_file(self, path):
        cache_path = path.replace('.wav', '.amazonlex')

        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)

        with open(path, 'rb') as input_audio:
            response = self._client.post_content(
                botName='barista',
                botAlias='$LATEST',
                userId=str(uuid.uuid4()),
                contentType='audio/l16; rate=16000; channels=1',
                accept='text/plain; charset=utf-8',
                inputStream=input_audio
            )

        result = dict(intent=response['intentName'],
                      slots={k: v for k, v in response['slots'].items() if v is not None},
                      transcript=response['inputTranscript'])

        with open(cache_path, 'w') as f:
            json.dump(result, f, indent=2)

        return result

    def __str__(self):
        return 'Amazon Lex'


class GoogleDialogflow(NLUEngine):
    def __init__(self, project_id):
        self._project_id = project_id

    def process_file(self, path):
        cache_path = path.replace('.wav', '.dialogflow')

        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)

        session_client = dialogflow.SessionsClient()

        session_id = os.path.basename(path)[0]

        session = session_client.session_path(self._project_id, session_id)

        with open(path, 'rb') as audio_file:
            input_audio = audio_file.read()

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
        return 'Google Dialogflow'


class IBMWatson(NLUEngine):
    def __init__(self, model_id, custom_id):
        self._model_id = model_id
        if custom_id is None:
            self._custom_id = create_language_model()
        else:
            self._custom_id = custom_id

    def process_file(self, path):
        cache_path = path.replace('.wav', '.watson')

        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return json.load(f)

        stt_service = SpeechToTextV1()
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

        nlu_service = NaturalLanguageUnderstandingV1(version='2018-03-16')

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


class PicovoiceRhino(NLUEngine):
    def __init__(self):
        self._library_path = _path('rhino/lib/linux/x86_64/libpv_rhino.so')
        self._model_path = _path('rhino/lib/common/rhino_params.pv')
        self._context_path = _path('rhino/resources/contexts/linux/coffee_maker_linux.rhn')

    def process_file(self, path):
        sys.path.append(_path('rhino/binding/python'))
        from rhino import Rhino

        rhino = Rhino(
            library_path=self._library_path,
            model_path=self._model_path,
            context_path=self._context_path)

        pcm, sample_rate = soundfile.read(path, dtype='int16')
        assert pcm.ndim == 1
        assert sample_rate == rhino.sample_rate

        is_finalized = False
        start_index = 0
        while start_index < (len(pcm) - rhino.frame_length) and not is_finalized:
            end_index = start_index + rhino.frame_length
            is_finalized = rhino.process(pcm[start_index: end_index])
            start_index = end_index

        if not is_finalized:
            result = None
        else:
            if rhino.is_understood():
                intent, slot_values = rhino.get_intent()
                result = dict(intent=intent, slots=slot_values)
            else:
                result = None

        return result

    def __str__(self):
        return 'Picovoice Rhino'
