import math 
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention with optional masking.

    Args:
        Q: Query tensor of shape (batch_size, num_heads, seq_len, d_k)
        K: Key tensor of shape (batch_size, num_heads, seq_len, d_k)
        V: Value tensor of shape (batch_size, num_heads, seq_len, d_v)
        mask: Optional boolean mask of shape (batch_size, num_heads, seq_len, seq_len).
              Positions where mask is True are ignored (filled with -inf before softmax).

    Returns:
        output:            Shape (batch_size, num_heads, seq_len, d_v)
        attention_weights: Shape (batch_size, num_heads, seq_len, seq_len)
    """

    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads:int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_k * num_heads)
        self.W_K = nn.Linear(d_model, d_k * num_heads)
        self.W_V = nn.Linear(d_model, d_v * num_heads)
        self.W_O = nn.Linear(d_v * num_heads, d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for multi-head attention.

        Args:
            Q: Query tensor of shape (batch_size, seq_len, d_model)
            K: Key tensor of shape (batch_size, seq_len, d_model)
            V: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Optional boolean mask of shape (batch_size, seq_len, seq_len)