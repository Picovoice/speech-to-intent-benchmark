# Speech to Intent Benchmark

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Picovoice/speech-to-intent-benchmark/blob/master/LICENSE)

This is a framework to benchmark the accuracy of Picovoice's speech-to-intent engine (a.k.a rhino) against other
natural language understanding engines. For more information regarding the engine refer to its repository directly [here](https://github.com/Picovoice/rhino).
This repository contains all data and code to reproduce the results. In this benchmark we evaluate the accuracy of engine for the context of voice
enabled coffee maker. You can listen to one of the sample commands [here](/data/speech/clean/81774d8e-7da7-4e9b-8cc3-33015b0ae0aa.wav).
In order to simulate the real-life situations we have tested in two noisy conditions (1) Cafe and (2) Kitchen. You can listen
to samples of noisy data [here](/data/misc/noisy1.wav) and [here](/data/misc/noisy2.wav).

Additionally we compare the accuracy of rhino with [Google Dialogflow](https://dialogflow.com/) and [Amazon Lex](https://aws.amazon.com/lex/).

# Data

The speech commands are crowd sourced from more than 50 unique speakers. Each speaker contributed about 10 different commands.
Collectively there are 619 commands used in this benchmark. Noise is downloaded from [Freesound](https://freesound.org/).

# Usage

Clone the directory and its submodules via

```bash
git clone --recurse-submodules https://github.com/Picovoice/speech-to-intent-benchmark.git
```

The repository grabs the latest version of rhino as a Git submodule under [rhino](/rhino). All data needed for this
benchmark including speech, noise, and labels are provided under [data](/data). Additionally the Dialogflow agents used
in this benchmark are exported [here](/data/dialogflow). The Amazon Lex bots used in this benchmark are exported [here](/data/amazonlex). The benchmark code is located under [benchmark](/benchmark).

The first step is to mix the clean speech data under [clean](/data/speech/clean) with noise. There are two types of noise
used for this benchmark (1) [cafe](/data/noise/cafe.wav) and (2) [kitchen](/data/noise/kitchen.wav). In order to create
noisy test data enter the following from the root of the repository in shell

```bash
python benchmark/mixer.py ${NOISE}
```

`${NOISE}` can be either `kitchen` or `cafe`.

Create accuracy results for running the noisy spoken commands through a NLU engine by running the following
```bash
python benchmark/benchmark.py --engine_type ${AN_ENGINE_TYPE} --noise ${NOISE}
```

The valid options for the `engine_type` parameter are: `AMAZON_LEX`, `GOOGLE_DIALOGFLOW`, and `PICOVOICE_RHINO`.

In order to run the noisy spoken commands through Dialogflow API, include your Google Cloud Platform credential path and Google Cloud Platform project ID like the following
```bash
python benchmark/benchmark.py --engine_type GOOGLE_DIALOGFLOW --gcp_credential_path ${GOOGLE_CLOUD_PLATFORM_CREDENTIAL_PATH} --gcp_project_id ${GOOGLE_CLOUD_PLATFORM_PROJECT_ID} --noise ${NOISE}
```

# Results

Below is the result of benchmark. Command Acceptance Probability (Accuracy) is defined as the probability of the engine
to correctly understand the speech command.

![](data/misc/result.png)