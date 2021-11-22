# Speech-to-Intent Benchmark

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Picovoice/speech-to-intent-benchmark/blob/master/LICENSE)

This framework benchmarks the accuracy of Picovoice's Speech-to-Intent engine, [Rhino](https://github.com/Picovoice/rhino).
It compares the accuracy of Rhino with cloud-based natural language understanding (NLU) offerings:

- [Google Dialogflow](https://dialogflow.com/)
- [Amazon Lex](https://aws.amazon.com/lex/)
- [Microsoft LUIS](https://www.luis.ai/)
- [IBM Watson](https://www.ibm.com/watson)

## Table of Contents

- [Speech-to-Intent Benchmark](#speech-to-intent-benchmark)
  - [Table of Contents](#table-of-contents)
  - [Results](#results)
  - [Data](#data)
  - [How to Reproduce?](#how-to-reproduce)

## Results

Command acceptance rate is the probability of an engine correctly understanding the spoken command. Below is the summary:

![](data/misc/result-summary.png)

The figure below depicts engines performance at each SNR:

![](data/misc/result.png)

## Data

The scenario under test is a voice-enabled coffee maker.

The speech commands are crowd sourced from more than 50 unique speakers. Each speaker contributed about 10 different commands.
Collectively there are 619 commands used in this benchmark. You can listen to one of the sample commands [here](/data/speech/clean/8a92c476-050d-4b5b-911e-24b661a5b69f.wav). In order to simulate the real-life situations we have tested in two noisy conditions (1) Cafe and (2) Kitchen. You can listen to samples of noisy data [here](/data/misc/noisy1.wav) and [here](/data/misc/noisy2.wav).Noise is downloaded from [Freesound](https://freesound.org/).

## How to Reproduce?

Clone the directory and its submodules via

```console
git clone --recurse-submodules https://github.com/Picovoice/speech-to-intent-benchmark.git
```

The first step is to mix the clean speech data under [clean](/data/speech/clean) with noise. There are two types of noise
used for this benchmark (1) [cafe](/data/noise/cafe.wav) and (2) [kitchen](/data/noise/kitchen.wav). In order to create
noisy test data enter the following from the root of the repository in shell

```console
python benchmark/mixer.py ${NOISE}
```

`${NOISE}` can be either `kitchen` or `cafe`.

Create accuracy results for running the noisy spoken commands through a NLU engine by running the following
```console
python benchmark/benchmark.py --engine_type ${AN_ENGINE_TYPE} --noise ${NOISE}
```

The valid options for the `engine_type` parameter are: `AMAZON_LEX`, `GOOGLE_DIALOGFLOW`, `IBM_WATSON`, `MICROSOFT_LUIS` and `PICOVOICE_RHINO`.

In order to run the noisy spoken commands through Dialogflow API, include your Google Cloud Platform credential path and Google Cloud Platform project ID like the following
```bash
python benchmark/benchmark.py --engine_type GOOGLE_DIALOGFLOW --gcp_credential_path ${GOOGLE_CLOUD_PLATFORM_CREDENTIAL_PATH} --gcp_project_id ${GOOGLE_CLOUD_PLATFORM_PROJECT_ID} --noise ${NOISE}
```

To run the noisy spoken commands through IBM Watson API, include your IBM Cloud credential path and your Natural Language Understanding model ID.  
If you already have a custom Speech to Text language model, include its ID using the argument `--ibm_custom_id`. Otherwise, a new custom language model
will be created for you.
```bash
python benchmark/benchmark.py --engine_type IBM_WATSON --ibm_credential_path ${IBM_CREDENTIAL_PATH} --ibm_model_id ${IBM_MODEL_ID} --noise ${NOISE}
```

Before running noisy spoken commands through Microsoft LUIS, add your LUIS credentials and Speech credentials into `/data/luis/credentials.env`.
