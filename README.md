# LENIE

**Node Importance Estimation Leveraging LLMs for Semantic Augmentation in Knowledge Graphs**

This repository provides the official implementation of **LENIE**, proposed in the paper *Node Importance Estimation Leveraging LLMs for Semantic Augmentation in Knowledge Graphs* (Knowledge-Based Systems, 2025).

LENIE enhances node semantic representations in Knowledge Graphs (KGs) using Large Language Models (LLMs), and improves the performance of downstream Node Importance Estimation (NIE) models without modifying their architectures.

---

## Introduction

Node Importance Estimation (NIE) aims to quantify the importance of nodes in a graph and plays a key role in many real-world applications. Existing NIE methods that utilize semantic information from KGs are often limited by **insufficient, missing, or inaccurate node descriptions**.

To address this issue, **LENIE** leverages LLMs as semantic augmenters. By integrating KG-derived triplets and original node descriptions into node-specific adaptive prompts, LENIE generates richer and more accurate augmented descriptions for nodes. These descriptions are then encoded as node embeddings to initialize downstream NIE models.

To the best of our knowledge, this work is the **first attempt to incorporate LLMs into the NIE task**.

Note: This repository contains the source code and datasets for the paper, with the downstream NIE models used in the paper available in their respective original repositories. 

---

## Method Overview

The LENIE framework consists of three main stages:

1. **Semantic Extraction from KGs**  
   - Collect one-hop triplets for each node  
   - Apply clustering-based triplet sampling to ensure semantic diversity

2. **LLM-based Semantic Augmentation**  
   - Construct node-specific adaptive prompts  
   - Generate augmented node descriptions using LLMs

3. **Downstream NIE**  
   - Encode augmented descriptions into embeddings  
   - Use the embeddings to initialize downstream NIE models

LENIE is model-agnostic and can be applied to various existing NIE models.

---

## Repository Structure
```text
LENIE/
├── data/
│   ├── raw/                  # Original knowledge graph data
│   ├── processed/            # Processed data
│
├── xxx_data_preprocess.py    # Triplet extraction and sampling
├── llm_combine.py            # LLM-based semantic augmentation
├── xxx_bert_embedding.py     # Text encoding for node embeddings
│
├── requirements.txt
└── README.md
```
---

## How to use
1. xxx_data_preprocess.py
2. llm_combine.py
3. xxx_bert_embedding.py
4. The semantically augmented embeddings from the previous steps are fed into the downstream NIE models for node embedding initialization.

---

## Citation
If you find this work helpful, please cite:

```text
@article{LIN2025114521,
title = {Node importance estimation leveraging LLMs for semantic augmentation in knowledge graphs},
journal = {Knowledge-Based Systems},
volume = {330},
pages = {114521},
year = {2025},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2025.114521},
url = {https://www.sciencedirect.com/science/article/pii/S0950705125015606},
author = {Xinyu Lin and Tianyu Zhang and Chengbin Hou and Jinbao Wang and Jianye Xue and Hairong Lv},
keywords = {Node importance estimation, Large language models, Knowledge graphs, Semantic augmentation, LLMs}
}
```
