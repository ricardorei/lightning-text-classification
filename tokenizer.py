# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer

from torchnlp.encoders import Encoder
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.encoders.text.text_encoder import TextEncoder


class Tokenizer(TextEncoder):
    """
    Wrapper arround BERT tokenizer.
    """

    def __init__(self, pretrained_model) -> None:
        self.enforce_reversible = False
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.itos = self.tokenizer.ids_to_tokens

    @property
    def unk_index(self) -> int:
        """Returns the index used for the unknown token."""
        return self.tokenizer.unk_token_id

    @property
    def bos_index(self) -> int:
        """Returns the index used for the begin-of-sentence token."""
        return self.tokenizer.cls_token_id

    @property
    def eos_index(self) -> int:
        """Returns the index used for the end-of-sentence token."""
        return self.tokenizer.sep_token_id

    @property
    def padding_index(self) -> int:
        """Returns the index used for padding."""
        return self.tokenizer.pad_token_id

    @property
    def vocab(self) -> list:
        """
        Returns:
            list: List of tokens in the dictionary.
        """
        return self.tokenizer.vocab

    @property
    def vocab_size(self) -> int:
        """
        Returns:
            int: Number of tokens in the dictionary.
        """
        return len(self.itos)

    def encode(self, sequence: str) -> torch.Tensor:
        """Encodes a 'sequence'.
        :param sequence: String 'sequence' to encode.

        :return: torch.Tensor with Encoding of the `sequence`.
        """
        sequence = TextEncoder.encode(self, sequence)
        return self.tokenizer(sequence, return_tensors="pt")["input_ids"][0]

    def batch_encode(self, sentences: list) -> (torch.Tensor, torch.Tensor):
        """
        :param iterator (iterator): Batch of text to encode.
        :param **kwargs: Keyword arguments passed to 'encode'.

        Returns
            torch.Tensor, torch.Tensor: Encoded and padded batch of sequences; Original lengths of
                sequences.
        """
        tokenizer_output = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            return_length=True,
            return_token_type_ids=False,
            return_attention_mask=False,
            truncation="only_first",
            max_length=512,
        )
        return tokenizer_output["input_ids"], tokenizer_output["length"]
