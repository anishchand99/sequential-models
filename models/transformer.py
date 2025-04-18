import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        nhead=2,
        num_layers=2,
        dropout=0.1,
        pad_idx=3,
        max_seq_length=512,
    ):
        """
        Transformer-based language model

        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimension of the token embeddings
            hidden_dim (int): Dimension of the hidden state
            nhead (int): No. of attehntion heads in attention
            num_layers (int): Number of Transformer encoder layers
            dropout (float): Dropout probability
            pad_idx (int): Index of the padding token
        """
        super(TransformerModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        self.max_seq_length = max_seq_length
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout, max_seq_length)
        # transformer block
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        # stack encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )
        # output layer
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    # Initialize weight glorot
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # Function to generate the causal mask
    def _generate_square_subsequent_mask(self, sz, device):
        # upper traiangle excluding diagonal for mask
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(self, input_ids, hidden=None):
        pad_mask = (input_ids == self.pad_idx).to(input_ids.device)
        seq_len = input_ids.size(1)
        causal_mask = self._generate_square_subsequent_mask(seq_len, input_ids.device)
        embedded = self.embedding(input_ids)
        embedded = self.pos_encoder(embedded)
        output = self.transformer_encoder(
            embedded, mask=causal_mask, src_key_padding_mask=pad_mask
        )
        output = self.fc(output)
        return output, None

    def predict_next_token(self, logits, temperature=1.0):
        # Apply temperature scaling with numerical stability
        scaled_logits = logits / temperature

        # Apply softmax to get probabilities
        probs = F.softmax(scaled_logits, dim=-1)

        # Sample from the probability distribution
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def generate(self, tokenizer, prompt, max_seq_length=100, temperature=1.0):
        self.eval()
        prompt_ids = tokenizer.encode(prompt, out_type=int)
        generated = prompt_ids.copy()
        for _ in range(max_seq_length):
            current_seq = generated[-self.max_seq_length :]
            input_tensor = torch.tensor(current_seq, dtype=torch.long).unsqueeze(0)
            device = next(self.parameters()).device
            input_tensor = input_tensor.to(device)
            output, _ = self.forward(input_tensor)
            next_token_logits = output[0, -1, :]
            next_token = self.predict_next_token(next_token_logits, temperature).item()
            next_token = max(0, min(next_token, self.vocab_size - 1))
            generated.append(next_token)
            if next_token == tokenizer.eos_id():
                break
        generated_text = tokenizer.decode(generated)
        return generated_text


# From Moodle:
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

        return self.dropout(x)
