import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.lstm import LSTMModel
from models.rnn import RNNModel
from models.transformer import TransformerModel
from text_dataset import TextDataset, collate_fn, create_dataloaders

# Configuration and Hyperparameters
# Paths to data and tokenizer model
TRAIN_FILE = "data/train.jsonl"
TEST_FILE = "data/test.jsonl"
TOKENIZER_PATH = "bpe_tokenizer.model"

# Hyperparameters
BATCH_SIZE = 128
SEQ_LENGTH = 128
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
PATIENCE = 5  # early stopping
VALIDATION_RATIO = 0.1  # 10% of training data = validation set

# Model hyperparameters
VOCAB_SIZE = 10000
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.1
PAD_IDX = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Create training/validation split
def create_train_validation_split(
    train_file, tokenizer_path, val_ratio=0.1, batch_size=128, seq_length=128
):
    """
    Create training and validation sets from the training data

    Args:
        train_file: Path to training JSONL file
        tokenizer_path: Path to tokenizer model
        val_ratio: Ratio of data to use for validation
        batch_size: Batch size for data loaders
        seq_length: Maximum sequence length

    Returns:
        train_loader, val_loader, tokenizer
    """

    print("Creating training/validation split...")

    # Load the tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    pad_id = tokenizer.pad_id()

    # Load all samples from training file
    all_samples = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            all_samples.append(json.loads(line))

    random.seed(42)
    random.shuffle(all_samples)

    # Split into training and validation
    split_idx = int(len(all_samples) * (1 - val_ratio))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(
        f"Split dataset: {len(train_samples)} training samples, {len(val_samples)} validation samples"
    )

    # temp files for the split datasets
    os.makedirs("data/temp", exist_ok=True)
    train_temp = "data/temp/temp_train.jsonl"
    val_temp = "data/temp/temp_val.jsonl"

    with open(train_temp, "w", encoding="utf-8") as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + "\n")

    with open(val_temp, "w", encoding="utf-8") as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + "\n")

    # Create datasets
    train_dataset = TextDataset(train_temp, tokenizer_path, seq_length)
    val_dataset = TextDataset(val_temp, tokenizer_path, seq_length)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader, tokenizer


# Model configurations
model_configs = [
    {
        "name": "RNN",
        "class": RNNModel,
        "checkpoint": "rnn_best_model.pt",
        "params": {
            "vocab_size": VOCAB_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "pad_idx": PAD_IDX,
        },
    },
    {
        "name": "LSTM",
        "class": LSTMModel,
        "checkpoint": "lstm_best_model.pt",
        "params": {
            "vocab_size": VOCAB_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "pad_idx": PAD_IDX,
        },
    },
    {
        "name": "Transformer",
        "class": TransformerModel,
        "checkpoint": "transformer_best_model.pt",
        "params": {
            "vocab_size": VOCAB_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "nhead": 2,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "pad_idx": PAD_IDX,
            "max_seq_length": SEQ_LENGTH,
        },
    },
]


# Training Function
def train_model(model_config, train_loader, val_loader):
    model_name = model_config["name"]
    checkpoint_path = model_config["checkpoint"]

    # Skip if model is already trained
    if os.path.exists(checkpoint_path):
        print(f"{model_name} checkpoint already exists, skipping training...")
        return [], []  # return empty to avoid error

    print(f"\n{'=' * 50}")
    print(f"Training {model_name} Model")
    print(f"{'=' * 50}\n")

    # Initialize the model
    model = model_config["class"](**model_config["params"])
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Adjust learning rate and add gradient clipping for Transformer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    # Training Loop
    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    # For timing
    total_training_time = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        model.train()

        train_loss = 0.0
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)

            optimizer.zero_grad()

            outputs, _ = model(input_ids)

            if model_name == "Transformer":
                # For transformer models: causal masking
                logits = outputs.contiguous().view(-1, VOCAB_SIZE)
                targets = target_ids.contiguous().view(-1)

                # Compute loss only on non-padding tokens
                mask = targets != PAD_IDX
                if mask.sum() > 0:
                    masked_logits = logits[mask]
                    masked_targets = targets[mask]
                    loss = criterion(masked_logits, masked_targets)
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                # For RNN/LSTM
                logits = outputs.view(-1, VOCAB_SIZE)
                targets = target_ids.view(-1)
                loss = criterion(logits, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids, target_ids = input_ids.to(device), target_ids.to(device)
                outputs, _ = model(input_ids)

                # Similar loss computation logic as in training
                if model_name == "Transformer":
                    logits = outputs.contiguous().view(-1, VOCAB_SIZE)
                    targets = target_ids.contiguous().view(-1)

                    mask = targets != PAD_IDX
                    if mask.sum() > 0:
                        masked_logits = logits[mask]
                        masked_targets = targets[mask]
                        loss = criterion(masked_logits, masked_targets)
                    else:
                        loss = torch.tensor(0.0, device=device)
                else:
                    logits = outputs.view(-1, VOCAB_SIZE)
                    targets = target_ids.view(-1)
                    loss = criterion(logits, targets)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)  # Adjust learning rate

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time

        print(
            f"Epoch {epoch}/{NUM_EPOCHS} | Time: {epoch_time:.2f}s | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # Early Stopping and Checkpoint save
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), checkpoint_path)
            print(f"========> Model checkpoint saved to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered for {model_name} model")
                break

    print(f"\n{model_name} Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Total training time: {total_training_time:.2f}s")

    # Plot and save the loss curves
    plot_loss_curves(model_name, train_losses, val_losses)

    # Return losses for potential plotting
    return train_losses, val_losses


# Plot loss curves function
def plot_loss_curves(model_name, train_losses, val_losses):
    """
    Plot training and validation loss curves and save to file

    Args:
        model_name: Name of model
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, "b-", label="Training Loss")
    plt.plot(epochs, val_losses, "r-", label="Validation Loss")

    plt.title(f"{model_name} Model - Learning Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Create directory for plots
    os.makedirs("plots", exist_ok=True)

    # Save the plots
    plt.savefig(
        f"plots/{model_name.lower()}_loss_curve.png", dpi=300, bbox_inches="tight"
    )
    print(f"Loss curve saved to plots/{model_name.lower()}_loss_curve.png")
    plt.close()


# Create plots for model comparison
def create_comparison_plots(model_results):
    """
    Create comparison plots for all models
    """
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(12, 7))

    for model_name, (_, val_losses, _) in model_results.items():
        if val_losses:
            epochs = range(1, len(val_losses) + 1)
            plt.plot(epochs, val_losses, label=f"{model_name}")

    plt.title("Model Comparison - Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig("plots/model_comparison_loss.png", dpi=300, bbox_inches="tight")
    print("Model comparison plot saved to plots/model_comparison_loss.png")
    plt.close()

    # Create training time comparison bar chart
    models = [name for name, (_, _, time) in model_results.items() if time > 0]
    times = [time for _, (_, _, time) in model_results.items() if time > 0]

    if models:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, times, color=["blue", "orange", "green"])

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}s",
                ha="center",
                va="bottom",
            )

        plt.title("Training Time Comparison")
        plt.xlabel("Model")
        plt.ylabel("Time (seconds)")
        plt.grid(axis="y")

        plt.savefig("plots/model_comparison_time.png", dpi=300, bbox_inches="tight")
        print("Training time comparison plot saved to plots/model_comparison_time.png")
        plt.close()


# Main execution
if __name__ == "__main__":
    print(f"Using device: {device}")
    print(
        f"Creating training/validation split with {VALIDATION_RATIO:.1%} validation ratio"
    )

    # Create the training/validation split
    train_loader, val_loader, tokenizer = create_train_validation_split(
        TRAIN_FILE,
        TOKENIZER_PATH,
        val_ratio=VALIDATION_RATIO,
        batch_size=BATCH_SIZE,
        seq_length=SEQ_LENGTH,
    )

    # Dictionary to store results for all models
    model_results = {}

    # Train all models
    for model_config in model_configs:
        model_name = model_config["name"]
        start_time = time.time()

        train_losses, val_losses = train_model(model_config, train_loader, val_loader)

        total_time = time.time() - start_time

        # Store results for comparison plots
        model_results[model_name] = (train_losses, val_losses, total_time)

    # Create plots
    create_comparison_plots(model_results)

    print("\nAll models trained!")

    # Clean up temp files
    print("Cleaning up temporary files")
    try:
        os.remove("data/temp/temp_train.jsonl")
        os.remove("data/temp/temp_val.jsonl")
        print("Temporary files removed!!!!")
    except:
        print("Could not remove temporary files.")
