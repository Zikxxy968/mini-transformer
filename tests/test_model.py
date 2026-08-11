import torch

from model import TransformerLM, generate_with_sampling


def test_transformer_lm_logits_shape():
    model = TransformerLM(vocab_size=64, d_model=32, num_heads=4, num_layers=2, max_seq_len=16)
    idx = torch.randint(0, 64, (2, 8))

    logits = model(idx)

    assert logits.shape == (2, 8, 64)


def test_generate_extends_sequence():
    model = TransformerLM(vocab_size=64, d_model=32, num_heads=4, num_layers=2, max_seq_len=16)
    idx = torch.randint(0, 64, (1, 5))

    out = generate_with_sampling(model, idx, max_new_tokens=3, top_k=8)

    assert out.shape == (1, 8)

