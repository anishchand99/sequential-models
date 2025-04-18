import json
import math

import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from models.lstm import LSTMModel
from models.rnn import RNNModel
from models.transformer import TransformerModel
from text_dataset import create_dataloaders


def calculate_perplexity(model, data_loader, device):
    """
    Calculate perplexity on a dataset

    Args:
        model: The language model
        data_loader: DataLoader with evaluation data
        device: Device compute

    Returns:
        perplexity: Perplexity value
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=3)  # ignore pad

    with torch.no_grad():
        for input_ids, target_ids in data_loader:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            output, _ = model(input_ids)

            # Calculate loss
            output = output.view(-1, output.size(-1))
            target_ids = target_ids.view(-1)

            # Ignore padding tokens when counting total tokens
            non_pad_mask = target_ids != 3
            num_tokens = non_pad_mask.sum().item()

            loss = criterion(output, target_ids)
            total_loss += loss.item()
            total_tokens += num_tokens

    # Calculate perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss)

    return perplexity


def calculate_bleu(model, tokenizer, test_data, device, num_samples=100):
    """
    Calculate BLEU score by comparing generated text to ground truth

    Args:
        model: The language model
        tokenizer: SentencePiece tokenizer
        test_data: List of test samples from the test set
        device: Device compute
        num_samples: Number of samples to use for BLEU calculation

    Returns:
        bleu_score: BLEU score
    """
    model.eval()
    bleu_scores = []
    smoother = SmoothingFunction().method1

    # Take a sample of test data
    sample_indices = np.random.choice(
        len(test_data), min(num_samples, len(test_data)), replace=False
    )

    for idx in sample_indices:
        sample = test_data[idx]
        prompt = sample["prompt"]
        completion = sample["completion"]

        # Generate text from prompt
        generated_text = model.generate(
            tokenizer, prompt, max_seq_length=50, temperature=0.8
        )

        # Extract only the generated completion, exlcude the prompt
        generated_completion = extract_completion(generated_text, prompt)

        reference = tokenizer.encode(completion, out_type=str)
        hypothesis = tokenizer.encode(generated_completion, out_type=str)

        # Calculate BLEU score for this sample
        if len(hypothesis) == 0:
            bleu = 0.0
        else:
            bleu = sentence_bleu([reference], hypothesis, smoothing_function=smoother)
        bleu_scores.append(bleu)

    # Return average BLEU score
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0


def extract_completion(generated_text, prompt):
    """
    Extract the completion part from generated text by removing the prompt

    Args:
        generated_text: The text generated
        prompt: The prompt given to the model

    """
    if generated_text.startswith(prompt):
        return generated_text[len(prompt) :].strip()

    tokens = generated_text.split()
    prompt_tokens = prompt.split()

    # Find where the completion starts
    for i in range(len(tokens)):
        if i >= len(prompt_tokens) or tokens[i] != prompt_tokens[i]:
            return " ".join(tokens[i:])

    # If not, return the whole text
    return generated_text


def generate_samples(model, tokenizer, prompts):
    """
    Generate text samples from a list of prompts

    Args:
        model: The language model
        tokenizer: SentencePiece tokenizer
        prompts: List of text prompts

    Returns:
        generated_texts: List of generated texts
    """
    model.eval()
    generated_texts = []

    for prompt in prompts:
        generated_text = model.generate(
            tokenizer, prompt, max_seq_length=100, temperature=0.8
        )
        generated_texts.append(generated_text)

    return generated_texts


if __name__ == "__main__":
    TEST_FILE = "data/test.jsonl"
    TOKENIZER_PATH = "bpe_tokenizer.model"
    BATCH_SIZE = 32
    SEQ_LENGTH = 128

    # Load tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(TOKENIZER_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data and create data loader
    _, test_loader, _ = create_dataloaders(
        "data/train.jsonl",
        TEST_FILE,
        TOKENIZER_PATH,
        batch_size=BATCH_SIZE,
        seq_length=SEQ_LENGTH,
    )

    test_data = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            test_data.append(json.loads(line))

    vocab_size = 10000

    # RNN Model
    rnn_model = RNNModel(vocab_size=vocab_size).to(device)
    rnn_model.load_state_dict(torch.load("rnn_best_model.pt"))

    # LSTM Model
    lstm_model = LSTMModel(vocab_size=vocab_size).to(device)
    lstm_model.load_state_dict(torch.load("lstm_best_model.pt"))

    # Transformer Model
    transformer_model = TransformerModel(
        vocab_size=vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        nhead=2,
        num_layers=2,
        dropout=0.1,
        pad_idx=3,
        max_seq_length=SEQ_LENGTH,
    ).to(device)
    transformer_model.load_state_dict(torch.load("transformer_best_model.pt"))

    # Calculate perplexity
    rnn_perplexity = calculate_perplexity(rnn_model, test_loader, device)
    lstm_perplexity = calculate_perplexity(lstm_model, test_loader, device)
    transformer_perplexity = calculate_perplexity(
        transformer_model, test_loader, device
    )

    print(f"Perplexity (lower is better):")
    print(f"RNN: {rnn_perplexity:.2f}")
    print(f"LSTM: {lstm_perplexity:.2f}")
    print(f"Transformer: {transformer_perplexity:.2f}")

    # Calculate BLEU score
    rnn_bleu = calculate_bleu(rnn_model, tokenizer, test_data, device)
    lstm_bleu = calculate_bleu(lstm_model, tokenizer, test_data, device)
    transformer_bleu = calculate_bleu(transformer_model, tokenizer, test_data, device)

    print(f"\nBLEU Score (higher is better):")
    print(f"RNN: {rnn_bleu:.4f}")
    print(f"LSTM: {lstm_bleu:.4f}")
    print(f"Transformer: {transformer_bleu:.4f}")

    # Sample text generation
    prompts = [
        "Which do you prefer? Dogs or cats?",
        "The meaning of life is",
        "Alice started to talk",
    ]

    print("\nRNN Responses:")
    for prompt, response in zip(
        prompts, generate_samples(rnn_model, tokenizer, prompts)
    ):
        print(f"Prompt: {prompt}")
        print(f"Response: {response}\n")

    print("\nLSTM Responses:")
    for prompt, response in zip(
        prompts, generate_samples(lstm_model, tokenizer, prompts)
    ):
        print(f"Prompt: {prompt}")
        print(f"Response: {response}\n")

    print("\nTransformer Responses:")
    for prompt, response in zip(
        prompts, generate_samples(transformer_model, tokenizer, prompts)
    ):
        print(f"Prompt: {prompt}")
        print(f"Response: {response}\n")
