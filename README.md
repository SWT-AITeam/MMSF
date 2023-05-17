# MMSF

This repository contians the source code and datasets for the paper: “Multi-Modal Semantics Fusion Model for Domain Relation Extraction via Information Bottleneck”.

## requirements

* python 3.8
* torch  1.7
* tqdm
* transformers
* numpy

### Clone and load BERT pretrained models

```
git clone https://github.com/SWT-AITeam/MMSF.git
mkdir MMSF/datasets/bert
cd MMSF/datasets/bert
sudo apt-get install git-lfs

## provide path of pretrained models
git clone https://huggingface.co/bert-base-cased
git clone https://huggingface.co/bert-base-uncased

cd bert-base-cased
git lfs pull
cd ..
cd bert-base-uncased
git lfs pull

```

### Run the Code

```
python train.py
```

