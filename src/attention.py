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

    def __init__(self, d_model: int, num_heads:int, d_k: int, d_v: int):
        


