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
  --batch_size                Batch size to be used.
  --accumulate_grad_batches   Accumulated gradients runs K small batches of \
                              size N before doing a backwards pass.
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
  --encoder_model             BERT encoder model to be used.
  --encoder_learning_rate     Encoder specific learning rate.
  --nr_frozen_epochs          Number of epochs we will keep the BERT parameters frozen.
  --learning_rate             Classification head learning rate.
  --dropout                   Dropout to be applied to the BERT embeddings.
  --train_csv                 Path to the file containing the train data.
  --dev_csv                   Path to the file containing the dev data.
  --test_csv                  Path to the file containing the test data.
  --loader_workers            How many subprocesses to use for data loading.
```

**Note:**
After BERT several BERT-like models were released. You can test different size models like Mini-BERT and DistilBERT which are much smaller.
- Mini-BERT only contains 2 encoder layers with hidden sizes of 128 features. Use it with the flag: `--encoder_model google/bert_uncased_L-2_H-128_A-2`
- DistilBERT contains only 6 layers with hidden sizes of 768 features. Use it with the flag: `--encoder_model distilbert-base-uncased`

Training command example:
```bash
python training.py \
    --gpus 0 \
    --batch_size 32 \
    --accumulate_grad_batches 1 \
    --loader_workers 8 \
    --nr_frozen_epochs 1 \
    --encoder_model google/bert_uncased_L-2_H-128_A-2 \
    --train_csv data/MP2_train.csv \
    --dev_csv data/MP2_dev.csv \
```

Testing the model:
```bash
python test.py --experiment experiments/version_{date} --test_data data/MP2_dev.csv
```

### Tensorboard:

Launch tensorboard with:
```bash
tensorboard --logdir="experiments/"
```

### Code Style:
To make sure all the code follows the same style we use [Black](https://github.com/psf/black).
