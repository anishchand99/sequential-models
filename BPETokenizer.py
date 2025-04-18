import json
import os
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Dataset

# reproducibility
torch.manual_seed(0)
np.random.seed(0)


# Merge all text files into a single corpus
def merge_text_files(data_dir, output_file):
    """
    Load all text files in the given directory and
    saved combined string in a file.

    :param data_dir: directory containing raw text files
    :param output_file: path to file where corpus will be saved
    """
    print(f"Loading raw text files from {data_dir}...")
    raw_text = ""

    # Get list of all text files
    text_files = list(Path(data_dir).glob("*.txt"))
    for file_name in text_files:
        with open(file_name, "r", encoding="utf-8") as file:
            raw_text += file.read() + "\n"

    # Write the combined text to the output file
    with open(output_file, "w", encoding="utf-8") as outputfile:
        outputfile.write(raw_text)

    print(f"Merged {len(text_files)} files into '{output_file}'")


# Function to train the SentencePiece BPE tokenizer
def train_bpe_tokenizer(corpus_file, vocab_size, model_prefix):
    # Train the tokenizer with special tokens
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        user_defined_symbols=",".join(["<bos>", "<eos>", "<pad>"]),
    )

    print(
        f"Tokenizer trained and saved as {model_prefix}.model and {model_prefix}.vocab"
    )
    return f"{model_prefix}.model"


# Function to load the JSONL data (training and test sets)
def load_jsonl_data(file_path):
    """
    Load data from a JSONL file.
    file_path: path to the JSONL file
    return: list of dictionaries containing the data
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


# Main execution
if __name__ == "__main__":
    VOCAB_SIZE = 10000
    CORPUS_FILE = "corpus.txt"
    MODEL_PREFIX = "bpe_tokenizer"
    # Check if the tokenizer already exists to avoid retraining
    if not os.path.exists(f"{MODEL_PREFIX}.model"):
        # Load and combine all raw text files
        raw_text = merge_text_files("data/raw", "corpus.txt")
        # Train the tokenizer
        tokenizer_path = train_bpe_tokenizer(CORPUS_FILE, VOCAB_SIZE, MODEL_PREFIX)
    else:
        print("Tokenizer already exists. Skipping training.")
        tokenizer_path = f"{MODEL_PREFIX}.model"

    # Load the trained tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)

    # TODO: REMOVE TEST
    # Test the tokenizer on a sample text from slides
    sample_text = "The quick brown fox jumped over the lazy dog"
    tokens = sp.encode(sample_text, out_type=str)
    ids = sp.encode(sample_text, out_type=int)

    print(f"\nSample text: {sample_text}")
    print(f"Tokenized: {tokens}")
    print(f"Token IDs: {ids}")
    # TODO: REMOVE TEST

