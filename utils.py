# -*- coding: utf-8 -*-
from datetime import datetime

import torch

from pytorch_lightning.logging import TestTubeLogger


def setup_testube_logger() -> TestTubeLogger:
    """ Function that sets the TestTubeLogger to be used. """
    try:
        job_id = os.environ["SLURM_JOB_ID"]
    except Exception:
        job_id = None

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")
    return TestTubeLogger(
        save_dir="experiments/",
        version=job_id + "_" + dt_string if job_id else dt_string,
        name="lightning_logs",
    )


def mask_fill(
    fill_value: float,
    tokens: torch.tensor,
    embeddings: torch.tensor,
    padding_index: int,
) -> torch.tensor:
    """
    Function that masks embeddings representing padded elements.
    :param fill_value: the value to fill the embeddings belonging to padded tokens.
    :param tokens: The input sequences [bsz x seq_len].
    :param embeddings: word embeddings [bsz x seq_len x hiddens].
    :param padding_index: Index of the padding token.
    """
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)
