import pytest

import torch

from aural_travels.model import TransformerNAT

@pytest.mark.parametrize("axial_attention", [False, True])
def test_transformer_nat_batch(axial_attention):
    model = TransformerNAT(hidden_size=128,
                           num_layers=3,
                           context_len=10,
                           grid_size=8,
                           axial_attention=axial_attention)
    model.eval()

    seq_len = 10 + 8*8

    with torch.no_grad():
        for b in [1, 4, 5, 16]:
            x = torch.randn((b, seq_len, 128))

            y1 = model(x)
            y2 = model(x)
            y3 = torch.cat([model(row[None, ...]) for row in x], dim=0)

            assert torch.allclose(y1, y2)
            assert torch.allclose(y1, y3)
