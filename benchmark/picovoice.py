import json
import os
import sys

import soundfile


def _path(x):
    return os.path.join(os.path.dirname(__file__), '../%s' % x)


sys.path.append(_path('rhino/binding/python'))

from rhino import Rhino

LIB_PATH = _path('rhino/lib/linux/x86_64/libpv_rhino.so')
MODEL_PATH = _path('rhino/lib/common/rhino_params.pv')
CONTEXT_PATH = _path('rhino/resources/contexts/linux/coffee_maker_linux.rhn')


def process_file(path):
    rhino = Rhino(
        library_path=LIB_PATH,
        model_file_path=MODEL_PATH,
        context_file_path=CONTEXT_PATH)

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
        intent = None
    else:
        if rhino.is_understood():
            intent, slot_values = rhino.get_intent()
            intent = dict(intent=intent, slots=slot_values)
        else:
            intent = None

    return intent


def process(folder):
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

            intent = process_file(os.path.join(folder, x))

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
    noise = sys.argv[1]

    for snr_db in [24, 21, 18, 15, 12, 9, 6]:
        print('%s %d db:' % (noise, snr_db))
        process(_path('data/speech/%s_%ddb' % (noise, snr_db)))
