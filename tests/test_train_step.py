import torch
import torch.nn as nn

from model import TransformerLM
from train import CustomAdamW, run_gradient_clipping


def test_single_train_step_updates_parameters():
    model = TransformerLM(vocab_size=64, d_model=32, num_heads=4, num_layers=2, max_seq_len=16)
    optimizer = CustomAdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))

    before = next(model.parameters()).detach().clone()
    logits = model(x)
    loss = criterion(logits.view(-1, 64), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    run_gradient_clipping(model.parameters(), max_norm=1.0)
    optimizer.step()

    after = next(model.parameters()).detach()
    assert not torch.equal(before, after)
