import torch 
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from attention import MultiHeadAttention

def test_multi_head_attention():
    d_model = 128
    num_heads = 8
    seq_len = 10
    batch_size = 4
    Q = torch.randn(batch_size, seq_len, d_model)
    K = torch.randn(batch_size, seq_len, d_model)
    V = torch.randn(batch_size, seq_len, d_model)

    attention = MultiHeadAttention(d_model, num_heads)
    output, attention_weights = attention(Q, K, V)
    assert output.shape == (batch_size, seq_len, d_model)
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len)

def test_bad_divisibility():
    try:
        MultiHeadAttention(10, 3)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

if __name__ == "__main__":
    test_multi_head_attention()
    test_bad_divisibility()
    print("All tests passed.")

