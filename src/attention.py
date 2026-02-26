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
    assert Q.shape[-1] == K.shape[-1], \
        f"Q and K must have the same last dimension (d_k), got {Q.shape[-1]} and {K.shape[-1]}"
    assert K.shape[-2] == V.shape[-2], \
        f"K and V must have the same seq_len, got {K.shape[-2]} and {V.shape[-2]}"

    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int):
        assert isinstance(d_model, int) and d_model > 0, \
            f"d_model must be a positive integer, got {d_model}"
        assert isinstance(num_heads, int) and num_heads > 0, \
            f"num_heads must be a positive integer, got {num_heads}"
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
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
        assert Q.dim() == 3, \
            f"Q must be 3D (batch_size, seq_len, d_model), got shape {Q.shape}"
        assert K.shape == Q.shape, \
            f"K shape {K.shape} must match Q shape {Q.shape}"
        assert V.shape == Q.shape, \
            f"V shape {V.shape} must match Q shape {Q.shape}"
        assert Q.shape[-1] == self.d_model, \
            f"Last dimension of Q ({Q.shape[-1]}) must equal d_model ({self.d_model})"
        if mask is not None:
            assert mask.dim() == 3, \
                f"mask must be 3D (batch_size, seq_len, seq_len), got shape {mask.shape}"
            mask = mask.unsqueeze(1)  # (batch, 1, seq, seq) — broadcasts across all heads

        Q = self._split_heads(self.W_Q(Q))
        K = self._split_heads(self.W_K(K))
        V = self._split_heads(self.W_V(V))

        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)

        return self.W_O(self._merge_heads(output)), attention_weights
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape (batch_size, seq_len, d_model) → (batch_size, num_heads, seq_len, d_k)
        so each head gets its own slice of the embedding dimension.

        Args:
            x: Shape (batch_size, seq_len, d_model)

        Returns:
            Shape (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)


    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        The purpose of this function is to merge the heads back into the original dimension.
        Merge the last dimension into (num_heads, d_k).
        Reshape to (batch_size, seq_len, d_model).
        Return the reshaped tensor.
        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, _num_heads, seq_len, _d_k = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
