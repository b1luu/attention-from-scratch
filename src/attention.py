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

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size = Q.shape[0]
        seq_len = Q.shape[1]
        d_model = Q.shape[2]
        num_heads = self.num_heads
        d_k = self.d_k
        d_v = self.d_v
        W_Q = self.W_Q(Q)
        W_K = self.W_K(K)
        W_V = self.W_V(V)

        Q = W_Q.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        K = W_K.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)
        V = W_V.view(batch_size, seq_len, num_heads, d_v).transpose(1, 2)

        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_O(output)
        return output, attention_weights
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the last dimension into (num_heads, d_k).
        Reshape to (batch_size, num_heads, seq_len, d_k).
        Return the reshaped tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        d_model = x.shape[2]
        num_heads = self.num_heads
        d_k = self.d_k
        return x.view(batch_size, seq_len, num_heads, d_k).transpose(1, 2)


    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        The purpose of this function is to merge the heads back into the original dimension.
        Merge the last dimension into (num_heads, d_k).
        Reshape to (batch_size, seq_len, d_model).
        Return the reshaped tensor.
        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, d_k)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        num_heads = self.num_heads
        d_k = self.d_k
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * d_k)

    def _project_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project the last dimension into (d_model).
        Return the projected tensor.
        Reshape to (batch_size, seq_len, d_model).
        The purpose of this function is to project the heads back to the original dimension.
        It is used to combine the heads back into the original dimension.
        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, d_k)
        """
        batch_size = x.shape[0]
    