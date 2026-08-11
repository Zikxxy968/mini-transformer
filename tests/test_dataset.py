import numpy as np

from train import CausalMemmapDataset


def test_causal_dataset_returns_shifted_targets(tmp_path):
    path = tmp_path / "data.bin"
    np.arange(20, dtype=np.int32).tofile(path)

    ds = CausalMemmapDataset(str(path), context_length=5)
    x, y = ds[0]

    assert x.tolist() == [0, 1, 2, 3, 4]
    assert y.tolist() == [1, 2, 3, 4, 5]
