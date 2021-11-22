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
python3 benchmark/mixer.py cafe
python3 benchmark/mixer.py kitchen
```

### Rhino

Signup for [Picovoice Console](https://console.picovoice.ai/) and get a free `AccessKey`. Then run the benchmark:

```console
python3 benchmark/benchmark.py \
--engine_type PICOVOICE_RHINO \
--access-key ${ACCESS_KEY} --noise cafe
python3 benchmark/benchmark.py \
--engine_type PICOVOICE_RHINO \
--access-key ${ACCESS_KEY} --noise kitchen
```

### Google Dialogflow

Signup for Google Cloud Platform. Then run the benchmark:

```console
python3 benchmark/benchmark.py \
--engine_type GOOGLE_DIALOGFLOW \
--gcp_credential_path {GOOGLE_CLOUD_PLATFORM_CREDENTIAL_PATH} \
--gcp_project_id ${GOOGLE_CLOUD_PLATFORM_PROJECT_ID} --noise cafe
python3 benchmark/benchmark.py \
--engine_type GOOGLE_DIALOGFLOW \
--gcp_credential_path {GOOGLE_CLOUD_PLATFORM_CREDENTIAL_PATH} \
--gcp_project_id ${GOOGLE_CLOUD_PLATFORM_PROJECT_ID} --noise kitchen
```

### Microsoft LUIS

Signup for Microsoft LUIS and add your credentials into [credentials.env](/data/luis/credentials.env). Then run the
benchmark:

```console
python3 benchmark/benchmark.py \
--engine_type MICROSOFT_LUIS \
--noise cafe
python3 benchmark/benchmark.py \
--engine_type MICROSOFT_LUIS \
--noise kitchen
```

### IBM Watson

Signup for IBM Watson. Then run the benchmark:

```console
python3 benchmark/benchmark.py \
--engine_type IBM_WATSON \
--ibm_credential_path ${IBM_CREDENTIAL_PATH} \
--ibm_model_id ${IBM_MODEL_ID} \
--noise cafe
python3 benchmark/benchmark.py \
--engine_type IBM_WATSON \
--ibm_credential_path ${IBM_CREDENTIAL_PATH} \
--ibm_model_id ${IBM_MODEL_ID} \
--noise kitchen
```
