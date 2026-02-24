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
    mask: Optional mask tensor of shape (batch_size, num_heads, seq_len, seq_len)

Returns:
    tuple[torch.Tensor, torch.Tensor]: Tuple containing the attention weights and the output tensor
"""

    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return attention_weights, output

