"""
Runs a model on a single node across N-gpus.
"""
import os

from bert_classifier import BERTClassifier
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from test_tube import HyperOptArgumentParser
from utils import setup_testube_logger


def main(hparams) -> None:
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = BERTClassifier(hparams)

    # ------------------------
    # 2 INIT EARLY STOPPING
    # ------------------------
    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta=0.0,
        patience=hparams.patience,
        verbose=True,
        mode=hparams.metric_mode,
    )
    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        logger=setup_testube_logger(),
        checkpoint_callback=True,
        early_stop_callback=early_stop_callback,
        default_save_path="experiments/",
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        use_amp=hparams.use_16bit,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        log_gpu_memory=hparams.log_gpu_memory,
        val_percent_check=hparams.val_percent_check,
    )

    # --------------------------------
    # 4 INIT MODEL CHECKPOINT CALLBACK
    # -------------------------------
    ckpt_path = os.path.join(
        trainer.default_save_path,
        trainer.logger.name,
        f"version_{trainer.logger.version}",
        "checkpoints",
    )
    # initialize Model Checkpoint Saver
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        save_top_k=hparams.save_top_k,
        verbose=True,
        monitor=hparams.monitor,
        period=1,
        mode=hparams.metric_mode,
    )
    trainer.checkpoint_callback = checkpoint_callback

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    parser = HyperOptArgumentParser(
        strategy="random_search",
        description="Minimalist BERT Classifier",
        add_help=False,
    )
    parser.add_argument("--seed", type=int, default=3, help="Training seed.")
    parser.add_argument(
        "--save_top_k",
        default=1,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )
    # Early Stopping
    parser.add_argument(
        "--monitor", default="val_acc", type=str, help="Quantity to monitor."
    )
    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=1,
        type=int,
        help="Number of epochs with no improvement \
            after which training will be stopped.",
    )
    parser.add_argument(
        "--min_epochs",
        default=1,
        type=int,
        help="Limits training to a minimum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=2,
        type=int,
        help="Limits training to a max number number of epochs",
    )

    # Batching
    parser.add_argument(
        "--batch_size", default=16, type=int, help="Batch size to be used."
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=1,
        type=int,
        help="Accumulated gradients runs K small batches of size N before \
            doing a backwards pass.",
    )

    # gpu args
    parser.add_argument("--gpus", type=int, default=2, help="How many gpus")
    parser.add_argument(
        "--distributed_backend",
        type=str,
        default="dp",
        help="Supports three options dp, ddp, ddp2",
    )
    parser.add_argument(
        "--use_16bit",
        dest="use_16bit",
        action="store_true",
        help="If true uses 16 bit precision",
    )
    parser.add_argument(
        "--log_gpu_memory",
        type=str,
        default=None,
        help="Uses the output of nvidia-smi to log GPU usage. \
            Might slow performance.",
    )  

    parser.add_argument(
        "--val_percent_check",
        default=1.0,
        type=float,
        help="If you don't want to use the entire dev set (for debugging or \
            if it's huge), set how much of the dev set you want to use with this flag.",
    )

    # each LightningModule defines arguments relevant to it
    parser = BERTClassifier.add_model_specific_args(parser)
    hparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)
