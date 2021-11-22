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

The speech data are crowd-sourced from more than 50 unique speakers. Each speaker contributed about ten different utterances.
Collectively there are 619 commands used in this benchmark. We test the engines in noisy conditions to simulate real-world situations. Noise is from [Freesound](https://freesound.org/).

## How to Reproduce?

Clone the repository:

```console
git clone --recurse-submodules https://github.com/Picovoice/speech-to-intent-benchmark.git
```

Mix the clean speech data with noise:

```console
python benchmark/mixer.py cafe
python benchmark/mixer.py kitchen
```

### Rhino

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
