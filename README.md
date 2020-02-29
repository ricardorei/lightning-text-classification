# Minimalist Implementation of a BERT Sentence Classifier

This repo is a minimalist implementation of a BERT Sentence Classifier.
The goal of this repo is to show how to combine 3 of my favourite libraries to supercharge your NLP research.

My favourite libraries:
- [PyTorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/)
- [Transformers](https://huggingface.co/transformers/index.html)
- [PyTorch-NLP](https://pytorchnlp.readthedocs.io/en/latest/index.html)


## Requirements:

This project uses Python 3.6

Create a virtual env with (outside the project folder):

```bash
virtualenv -p python3.6 sbert-env
source sbert-env/bin/activate
```

Install the requirements (inside the project folder):
```bash
pip install -r requirements.txt
```

## Getting Started:

### Train:
```bash
python training.py
```

Available commands:

Training arguments:
```bash
optional arguments:
  --seed                      Training seed.
  --distributed_backend       Supports three options: dp
  --use_16bit                 If true uses 16 bit precision
  --batch_size                Batch size to be used.
  --accumulate_grad_batches   Accumulated gradients runs K small batches of \
                              size N before doing a backwards pass.
  --log_gpu_memory            Uses the output of nvidia-smi to log GPU usage. \
                              Might slow performance.
  --val_percent_check         If you dont want to use the entire dev set, set \
                              how much of the dev set you want to use with this flag.      
```

Early Stopping/Checkpoint arguments:
```bash
optional arguments:
  --metric_mode             If we want to min/max the monitored quantity.
  --min_epochs              Limits training to a minimum number of epochs
  --max_epochs              Limits training to a max number number of epochs
  --save_top_k              The best k models according to the quantity \
                            monitored will be saved.
```

Model arguments:

```bash
optional arguments:
  --encoder_learning_rate     Encoder specific learning rate.
  --learning_rate             Classification head learning rate.
  --class_weights             Weights for each of the classes we want to tag.
  --warmup_steps              Scheduler warmup steps.
  --dropout                   Dropout to be applied to the BERT embeddings.
  --train_csv                 Path to the file containing the train data.
  --dev_csv                   Path to the file containing the dev data.
  --test_csv                  Path to the file containing the test data.
  --loader_workers            How many subprocesses to use for data loading.
```

Training command example:
```bash
python training.py \
    --gpus 2 \
    --distributed_backend dp \
    --batch_size 16 \
    --loader_workers 12 \
    --nr_frozen_epochs 1
```

### Tensorboard:

Launch tensorboard with:
```bash
tensorboard --logdir="experiments/lightning_logs/"
```

### Code Style:
To make sure all the code follows the same style we use [Black](https://github.com/psf/black).