import torch

from model import TransformerLM
from train import CustomAdamW, save_checkpoint


def test_checkpoint_save_and_load(tmp_path):
    model = TransformerLM(vocab_size=64, d_model=32, num_heads=4, num_layers=2, max_seq_len=16)
    optimizer = CustomAdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "epoch_1.pt"
    config = {
        "vocab_size": 64,
        "context_length": 16,
        "num_layers": 2,
        "num_heads": 4,
        "d_model": 32,
    }

    save_checkpoint(str(path), model, optimizer, iteration=1, epoch=1, config=config)
    ckpt = torch.load(path, map_location="cpu")
    reloaded = TransformerLM(vocab_size=64, d_model=32, num_heads=4, num_layers=2, max_seq_len=16)
    reloaded.load_state_dict(ckpt["model"])

    idx = torch.randint(0, 64, (1, 4))
    assert reloaded(idx).shape == (1, 4, 64)
