# GPT2

This is my attempt at recreating GPT2. Following [Karpathy's wonderful video](https://www.youtube.com/watch?v=l8pRSuU81PU&t=14190s), I replicated the architecture and trained the model on a fineweb_edu dataset.

## Dependencies
```
pip install torch numpy transformers datasets tiktoken tqdm
```

## Quick Start 

#### Prepare the Data
To tokenize and split the data into train and validation, run fineweb.py.
```bash
python3 fineweb.py
```
This fetches the data from the fineweb_edu10B dataset and populates the directory with the necessary data to train the model.

#### Training GPT2
To train gpt2, you'll need a minimum 4 gpus. I recommend 8. Run:
```bash
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

#### After Training
In the log directory, you will see you're trained models corresponding to differet points in training and also an output log.

## To Dos
- fine tune on tinyshakespeare
- add script for using the created model

