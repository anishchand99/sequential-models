import json

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer_path, sequence_length=128):
        """
        Create a text dataset for PyTorch Dataset that handles our jsonl prompts+completions
        Args:
          filepath: path to the jsonl data file
          tokenizer_path: path to the trained SentencePiece tokenizer model
          sequence_length: maximum sequence length we want to allow
        """
        self.sequence_length = sequence_length
        # Load the tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        self.samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # we are using Casual Language Modeling, prompts and completions treated the same way
                text = item["prompt"] + " " + item["completion"]
                # tokenize the full prompt + completion (truncate at max seq length)
                token_ids = self.tokenizer.encode(text, out_type=int)[:sequence_length]
                # make sure we don't have any overly short samples
                if len(token_ids) < 2:
                    continue
                # append tokenized sample to list
                self.samples.append(token_ids)

        print(f"Loaded {len(self.samples)} samples from {filepath}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """get an format a sample at given idx."""
        tokens = self.samples[idx]
        # Ensure tokens are within valid range for the vocabulary
        # Ensure tokens are within valid range (0 to vocab_size-1)
        tokens = [t for t in tokens if 0 <= t < self.tokenizer.get_piece_size()]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids


def collate_fn(batch, padding_value=3):
    """Ensure batch is appropriately sized and padded for efficient training
    args:
         batch: batch from DataLoader
    """
    input_batch, target_batch = zip(*batch)
    input_batch = nn.utils.rnn.pad_sequence(
        input_batch, batch_first=True, padding_value=padding_value
    )
    target_batch = nn.utils.rnn.pad_sequence(
        target_batch, batch_first=True, padding_value=padding_value
    )
    # TODO:
    # vocab_size = 10000
    # target_batch = torch.clamp(target_batch, 0, vocab_size-1)
    return input_batch, target_batch


def create_dataloaders(
    train_file, validation_file, tokenizer_path, batch_size=128, seq_length=128
):
    """
    create DataLoaders for training and validation
    Args:
        train_file: path to training JSONL file
        validation_file: path to validation JSONL file
        tokenizer_path: path to tokenizer model
        batch_size: batch size for training
        seq_length: max seq length

    Returns:
        train_loader, validation_loader, tokenizer
    """
    # Load the tokenizer to get pad_id
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    pad_id = tokenizer.pad_id()

    # Create datasets
    train_dataset = TextDataset(train_file, tokenizer_path, seq_length)
    validation_dataset = TextDataset(validation_file, tokenizer_path, seq_length)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, validation_loader, tokenizer
