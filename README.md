# Speech-to-Intent Benchmark

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Picovoice/speech-to-intent-benchmark/blob/master/LICENSE)

Made in Vancouver, Canada by [Picovoice](https://picovoice.ai)

[![Twitter URL](https://img.shields.io/twitter/url?label=%40AiPicovoice&style=social&url=https%3A%2F%2Ftwitter.com%2FAiPicovoice)](https://twitter.com/AiPicovoice)
[![YouTube Channel Views](https://img.shields.io/youtube/channel/views/UCAdi9sTCXLosG1XeqDwLx7w?label=YouTube&style=social)](https://www.youtube.com/channel/UCAdi9sTCXLosG1XeqDwLx7w)

This framework benchmarks the accuracy of Picovoice's Speech-to-Intent engine, [Rhino](https://github.com/Picovoice/rhino).
It compares the accuracy of Rhino with:

- [Amazon Lex](https://aws.amazon.com/lex/)
- [Google Dialogflow](https://dialogflow.com/)
- [IBM Watson](https://www.ibm.com/watson)
- [Microsoft LUIS](https://www.luis.ai/)

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
Collectively there are 619 commands used in this benchmark. We test the engines in noisy conditions to simulate real-world
situations. Noise is from [Freesound](https://freesound.org/).

## How to Reproduce?

Clone the repository:

```console
git clone https://github.com/Picovoice/speech-to-intent-benchmark.git
```

Mix the clean speech data with noise:

```console
python3 src/mix.py cafe
python3 src/mix.py kitchen
```

Get the usage message:

```console
python3 src/bench.py --help
```

Then run the script for each engine.
