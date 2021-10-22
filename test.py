"""
Tests model.
"""
import os
from argparse import ArgumentParser, Namespace

import pandas as pd
import yaml
from sklearn.metrics import classification_report
from tqdm import tqdm

from classifier import Classifier


def load_model_from_experiment(experiment_folder: str):
    """Function that loads the model from an experiment folder.
    :param experiment_folder: Path to the experiment folder.
    Return:
        - Pretrained model.
    """
    hparams_file = experiment_folder + "/hparams.yaml"
    hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)

    checkpoints = [
        file
        for file in os.listdir(experiment_folder + "/checkpoints/")
        if file.endswith(".ckpt")
    ]
    checkpoint_path = experiment_folder + "/checkpoints/" + checkpoints[-1]
    model = Classifier.load_from_checkpoint(
        checkpoint_path, hparams=Namespace(**hparams)
    )
    # Make sure model is in prediction mode
    model.eval()
    model.freeze()
    return model


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Minimalist Transformer Classifier", add_help=True
    )
    parser.add_argument(
        "--experiment",
        required=True,
        type=str,
        help="Path to the experiment folder.",
    )
    parser.add_argument(
        "--test_data",
        required=True,
        type=str,
        help="Path to the test data.",
    )
    hparams = parser.parse_args()
    hparams = parser.parse_args()
    print("Loading model...")
    model = load_model_from_experiment(hparams.experiment)
    # print(model)

    testset = pd.read_csv(hparams.test_data).to_dict("records")
    predictions = [
        model.predict(sample)
        for sample in tqdm(testset, desc="Testing on {}".format(hparams.test_data))
    ]

    y_pred = [o["predicted_label"] for o in predictions]
    y_true = [s["label"] for s in testset]
    print(classification_report(y_true, y_pred))
