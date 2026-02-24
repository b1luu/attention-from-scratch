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