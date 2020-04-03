"""
Runs a script to interact with a model using the shell.
"""
import os

import pandas as pd

from bert_classifier import BERTClassifier
from test_tube import HyperOptArgumentParser


def load_model_from_experiment(experiment_folder: str):
    """ Function that loads the model from an experiment folder.
    :param experiment_folder: Path to the experiment folder.
    Return:
        - Pretrained model.
    """
    tags_csv_file = experiment_folder + "/meta_tags.csv"
    tags = pd.read_csv(tags_csv_file, header=None, index_col=0, squeeze=True).to_dict()
    checkpoints = [
        file
        for file in os.listdir(experiment_folder + "/checkpoints/")
        if file.endswith(".ckpt")
    ]
    checkpoint_path = experiment_folder + "/checkpoints/" + checkpoints[-1]
    model = BERTClassifier.load_from_metrics(
        weights_path=checkpoint_path, tags_csv=tags_csv_file
    )
    # Make sure model is in prediction mode
    model.eval()
    model.freeze()
    return model


if __name__ == "__main__":
    parser = HyperOptArgumentParser(
        description="Minimalist BERT Classifier", add_help=True
    )
    parser.add_argument(
        "--experiment", required=True, type=str, help="Path to the experiment folder.",
    )
    hparams = parser.parse_args()
    print("Loading model...")
    model = load_model_from_experiment(hparams.experiment)
    print(model)

    while 1:
        print("Please write a movie review or quit to exit the interactive shell:")
        # Get input sentence
        input_sentence = input("> ")
        if input_sentence == "q" or input_sentence == "quit":
            break
        prediction = model.predict(sample={"text": input_sentence})
        print(prediction)
