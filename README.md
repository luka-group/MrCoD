# Multi-hop Evidence Retrieval for Cross-document Relation Extraction

Authors: Keming Lu, I-Hung Hsu, Wenxuan Zhou, Mingyu Derek Ma, Muhao Chen

:tada: This work is accepted by Findings of ACL2023 :tada: [Paper](https://aclanthology.org/2023.findings-acl.657.pdf)

## Overview

This repository provides codes and scripts for MR.COD (Multi-hop evidence retrieval for Cross-document relation extraction), which is a multi-hop evidence retrieval method based on evidence path mining and ranking.

## Key Idea of Mr.CoD

Relation Extraction (RE) has been extended to cross-document scenarios because many relations are not simply described in a single document. This inevitably brings the challenge of efficient open-space evidence retrieval to support the inference of cross-document relations, along with the challenge of multi-hop reasoning on top of entities and evidence scattered in an open set of documents. To combat these challenges, we propose MR.COD (Multi-hop evidence retrieval for Cross-document relation extraction), which is a multi-hop evidence retrieval method based on evidence path mining and ranking. We explore multiple variants of retrievers to show evidence retrieval is essential in cross-document RE. We also propose a contextual dense retriever for this setting. Experiments on CodRED show that evidence retrieval with MR.COD effectively acquires crossdocument evidence and boosts end-to-end RE performance in both closed and open settings

## Setup

This repo, especially training and inference codes in `BERT+ATT_scripts` for training cross-document RE models are developed based on [CodRED](https://github.com/thunlp/CodRED). Please kindly follow requirements of CodRED to set up experiment environments and download raw data.

We sincerely appreciate authors of CodRED for providing such great codes and data.

The notebook scripts under `data_scripts` contains all scripts for data processing described in the paper.

- data\_preprocessing(main).ipynb: first-step preprocessing script of CodRED raw data, including caching corpus into redis, processing open-setting queries, and evidence path mining (Section 3.2).

- extract\_question\_and\_evaluation\_for\_dpr(main).ipynb: the script to extract DPR inference and finetuning data from raw data of CodRED.

- passage\_retrieval(main).ipynb: extract evidence path for each query in CodRED based on the DPR outputs.

- path\_ranking (main).ipynb: rank retrieved evidence path based on semantic correlations.

## Training

Training scripts locate in the directory `BERT+ATT\_scripts`, which is developed based on the `run_blend.py` script in [CodRED](https://github.com/thunlp/CodRE).

`run.sh` is the script to launch training with the index file generated by data scripts and default training set.
`run_infer.sh` is the script to launch inference with closed/open set index file and data.

## Contact

Keming Lu (keminglu@usc.edu)

## Cite

```
@inproceedings{lu-etal-2023-multi,
    title = "Multi-hop Evidence Retrieval for Cross-document Relation Extraction",
    author = "Lu, Keming  and
      Hsu, I-Hung  and
      Zhou, Wenxuan  and
      Ma, Mingyu Derek  and
      Chen, Muhao",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.657",
    doi = "10.18653/v1/2023.findings-acl.657",
    pages = "10336--10351",
    abstract = "Relation Extraction (RE) has been extended to cross-document scenarios because many relations are not simply described in a single document.This inevitably brings the challenge of efficient open-space evidence retrieval to support the inference of cross-document relations,along with the challenge of multi-hop reasoning on top of entities and evidence scattered in an open set of documents.To combat these challenges, we propose Mr.Cod (Multi-hop evidence retrieval for Cross-document relation extraction), which is a multi-hop evidence retrieval method based on evidence path mining and ranking.We explore multiple variants of retrievers to show evidence retrieval is essential in cross-document RE.We also propose a contextual dense retriever for this setting.Experiments on CodRED show that evidence retrieval with Mr.Cod effectively acquires cross-document evidence and boosts end-to-end RE performance in both closed and open settings.",
}
```
