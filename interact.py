"""
Runs a script to interact with a model using the shell.
"""
import os
from argparse import ArgumentParser, Namespace

import pandas as pd
import yaml

from classifier import Classifier


def load_model_from_experiment(experiment_folder: str):
    """ Function that loads the model from an experiment folder.
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
        "--experiment", required=True, type=str, help="Path to the experiment folder.",
    )
    hparams = parser.parse_args()
    print("Loading model...")
    model = load_model_from_experiment(hparams.experiment)
    print(model)

    while 1:
        print("Please write a sentence or quit to exit the interactive shell:")
        # Get input sentence
        input_sentence = input("> ")
        if input_sentence == "q" or input_sentence == "quit":
            break
        prediction = model.predict(sample={"text": input_sentence})
        print(prediction)
