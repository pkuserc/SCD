<div align="center">

**Language Drift in Multilingual Retrieval-Augmented Generation: Characterization and Decoding-Time Mitigation** 

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/pdf/2511.09984)
[![Venue](https://img.shields.io/badge/AAAI-2026-blue.svg)](https://aaai.org/conference/aaai/aaai-26/)
[![Task](https://img.shields.io/badge/Task-Multilingual%20RAG-purple.svg)](#overview)

**AAAI 2026 Oral**

 <a href="https://deepblue666.github.io/">Bo Li</a>, Zhenghua Xu, Rui Xie

</div>

---

## Overview

Multilingual retrieval-augmented generation (RAG) allows large language models to answer knowledge-intensive questions by using retrieved documents as external evidence. However, when the language of the retrieved evidence differs from the language of the user query or in-context exemplars, the model may generate responses in an unintended language. This phenomenon is referred to as **language drift**.

This issue becomes especially visible in reasoning-heavy generation, such as chain-of-thought decoding, where intermediate steps can further amplify language instability. Our work systematically studies language drift across multiple multilingual QA datasets, languages, and model backbones, and shows that the problem is not simply caused by comprehension failure. Instead, it is strongly related to decoder-level behavior, where dominant token distributions, especially English, can override the intended target language.

To mitigate this, we propose **Soft Constrained Decoding (SCD)**, a lightweight and training-free decoding strategy that softly penalizes non-target-language tokens during generation. SCD is model-agnostic and can be integrated into standard generation pipelines without modifying model architecture or requiring additional training data. Experiments on three multilingual datasets show consistent improvements in language alignment and downstream task performance.

### Key Features

- **Training-free** decoding-time mitigation
- **Model-agnostic** and easy to integrate
- Focuses on **language alignment** in multilingual RAG
- Includes released multilingual versions of three QA datasets
- Suitable for analysis and follow-up research on multilingual reasoning and generation drift

---

## Repository Structure

The current repository contains the following main files:

```text
.
├── README.md
├── SCD.py
├── data generation.py
├── dureader_MultiLang_1000.json
├── hotpotqa_MultiLang_1000.json
└── musique_MultiLang_1000.json
```

### File Description

- `SCD.py`  
  Main script containing the decoding-time mitigation logic for SCD.

- `data generation.py`  
  Script related to multilingual data construction / generation.

- `dureader_MultiLang_1000.json`  
  Multilingual DuReader-based dataset.

- `hotpotqa_MultiLang_1000.json`  
  Multilingual HotpotQA-based dataset.

- `musique_MultiLang_1000.json`  
  Multilingual MuSiQue-based dataset.

---

## Method Summary

SCD addresses multilingual generation drift at decoding time.

Instead of retraining the model or introducing an additional controller, SCD adjusts the decoding distribution by softly discouraging tokens that are inconsistent with the intended target language. In this way, the method keeps generation close to the desired language while preserving the flexibility of open-ended reasoning.

This design makes SCD:

- simple to implement
- lightweight in inference
- compatible with standard autoregressive generation
- applicable to multilingual RAG settings with cross-lingual retrieval interference

---

## Datasets

This repository releases multilingual versions of three QA datasets for multilingual RAG research:

- **DuReader**
- **HotpotQA**
- **MuSiQue**

The released files are:

```text
dureader_MultiLang_1000.json
hotpotqa_MultiLang_1000.json
musique_MultiLang_1000.json
```

These files can be used for:
- multilingual RAG evaluation
- language drift analysis
- decoding-time intervention experiments
- follow-up work on multilingual reasoning and language control

---

## Usage

The repository is currently organized as a lightweight release of the core code and datasets.

### Basic Workflow

1. Prepare your multilingual RAG input data
2. Load a target model in `SCD.py`
3. Configure the target language and decoding settings
4. Run generation with the SCD decoding processor
5. Evaluate output language consistency and task performance on the released datasets

### Notes

- The main decoding implementation is in `SCD.py`.
- The current code is most suitable for research use and follow-up adaptation.
- Depending on your environment, you may need to adjust model paths, device settings, and tokenizer/model loading code before running experiments.

---

## Example Research Scope

This repository is especially useful for studying:

- output language drift in multilingual RAG
- cross-lingual interference during reasoning
- decoding-time language control
- multilingual chain-of-thought stability
- lightweight mitigation strategies without retraining

---

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{DBLP:conf/aaai/LiXX26,
  author       = {Bo Li and
                  Zhenghua Xu and
                  Rui Xie},
  editor       = {Sven Koenig and
                  Chad Jenkins and
                  Matthew E. Taylor},
  title        = {Language Drift in Multilingual Retrieval-Augmented Generation: Characterization
                  and Decoding-Time Mitigation},
  booktitle    = {Fortieth {AAAI} Conference on Artificial Intelligence, Thirty-Eighth
                  Conference on Innovative Applications of Artificial Intelligence,
                  Sixteenth Symposium on Educational Advances in Artificial Intelligence,
                  {AAAI} 2026, Singapore, January 20-27, 2026},
  pages        = {31519--31526},
  publisher    = {{AAAI} Press},
  year         = {2026},
}
```
