import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2,
        dropout=0.1,
        pad_idx=3,
    ):
        """
        Vanilla RNN-based language model

        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimension of the token embeddings
            hidden_dim (int): Dimension of the hidden state in the RNN
            num_layers (int): Number of RNN layers
            dropout (float): Dropout probability
            pad_idx (int): Index of the padding token
        """
        super(RNNModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        # Embedding layer maps token IDs to token embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # RNN layers
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, hidden=None):
        """
        compute model output logits given a sequence

        Args:
            input_ids: sequence of input token IDs, Tensor of shape (batch_size, seq_len)
            hidden: Initial hidden state

        Returns:
            output: Tensor shape [batch_size, seq_len, vocab_size]
            hidden: Final hidden state
        """
        # Convert token IDs to embeddings
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)

        # Pass through RNN
        output, hidden = self.rnn(embedded, hidden)
        # Apply dropout to RNN output
        output = self.dropout(output)

        # Apply output layer to each timestep
        output = self.fc(output)

        return output, hidden

    def predict_next_token(self, logits, temperature=1.0):
        """
        predict next token based on the predicted input_ids

        Args:
            input_ids: input sequence toke ids
            temperature: Temperature for sampling (higher = more randomness)

        Returns:
            next_token: Sampled token ID
        """
        # Apply temperature scaling with numerical stability
        scaled_logits = logits / temperature

        # Apply softmax to get probabilities
        probs = F.softmax(scaled_logits, dim=-1)

        # Sample from the probability distribution
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def generate(self, tokenizer, prompt, max_seq_length=100, temperature=1.0):
        """
        Generate text from a prompt

        Args:
            tokenizer: SentencePiece tokenizer
            prompt: Text prompt to start generation
            max_seq_length: Maximum length of the generated sequence
            temperature: Temperature for sampling (higher = more randomness)

        Returns:
            generated_text: Generated text as a string
        """
        self.eval()
        # Tokenize the prompt
        prompt_ids = tokenizer.encode(prompt, out_type=int)
        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0)

        device = next(self.parameters()).device
        input_tensor = input_tensor.to(device)

        # Initial hidden state is None
        hidden = None

        # Initialize list for storing generated tokens
        generated = []

        # Generate tokens until EOS or max length
        with torch.no_grad():
            for _ in range(max_seq_length):
                # Get predictions for last token
                output, hidden = self.forward(input_tensor, hidden)
                next_token_logits = output[0, -1, :]

                # Sample the next token
                next_token = self.predict_next_token(next_token_logits, temperature)
                next_token = next_token.item()

                # Append to generated tokens
                generated.append(next_token)

                if next_token == tokenizer.eos_id():
                    break

                # Update input_tensor
                input_tensor = torch.tensor([[next_token]], dtype=torch.long).to(device)

        # Convert tokens back to text
        generated_text = tokenizer.decode(generated)

        return generated_text
