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

def test_attention_weights_sum_to_one():
    # After softmax, each token's attention distribution must sum to 1.
    # This is the core property of a probability distribution.
    model = MultiHeadAttention(d_model=16, num_heads=2)
    Q = torch.randn(2, 4, 16)
    _, attention_weights = model(Q, Q, Q)
    sums = attention_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

def test_single_head():
    # Edge case: num_heads=1 means the model does no splitting at all.
    # Output should still be the correct shape.
    model = MultiHeadAttention(d_model=16, num_heads=1)
    Q = torch.randn(1, 5, 16)
    output, attention_weights = model(Q, Q, Q)
    assert output.shape == (1, 5, 16)
    assert attention_weights.shape == (1, 1, 5, 5)

def test_mask_blocks_attention():
    # Masked positions should receive ~0 attention weight.
    # We mask every position except the first, then check that
    # tokens 1+ are ignored by looking at the first query's weights.
    model = MultiHeadAttention(d_model=16, num_heads=2)
    Q = torch.randn(1, 4, 16)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    mask[:, :, 1:] = True  # block all keys except position 0
    _, attention_weights = model(Q, Q, Q, mask)
    blocked_weights = attention_weights[:, :, :, 1:]
    assert torch.allclose(blocked_weights, torch.zeros_like(blocked_weights), atol=1e-6)

def test_gradients_flow():
    # Verify that loss.backward() runs without error and produces gradients.
    # If any operation broke the computation graph, grad would be None.
    model = MultiHeadAttention(d_model=16, num_heads=2)
    Q = torch.randn(2, 4, 16, requires_grad=True)
    output, _ = model(Q, Q, Q)
    loss = output.sum()
    loss.backward()
    assert Q.grad is not None

def test_bad_divisibility():
    try:
        MultiHeadAttention(10, 3)
        assert False, "Should have raised AssertionError"
    except AssertionError:
        pass

if __name__ == "__main__":
    test_multi_head_attention()
    test_attention_weights_sum_to_one()
    test_single_head()
    test_mask_blocks_attention()
    test_gradients_flow()
    test_bad_divisibility()
    print("All tests passed.")

