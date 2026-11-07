import torch
from src.models.dkt import DKT
from src.models.neurotrace import UAGKT


def test_dkt_shape():
    m = DKT(8)
    c = torch.randint(0, 8, (4, 10))
    r = torch.randint(0, 2, (4, 10)).float()
    out = m(c, r)
    assert out.shape == (4, 10)


def test_uagkt_forward_and_uncertainty():
    K, T, B = 8, 12, 3
    m = UAGKT(K)
    A = torch.eye(K); U = torch.zeros(K, K)
    c = torch.randint(0, K, (B, T))
    r = torch.randint(0, 2, (B, T)).float()
    out = m(c, r, A, U)
    assert out["logits"].shape == (B, T)
    assert out["mastery"].shape == (B, T, K)

    res = m.predict_with_uncertainty(c, r, A, U, n_samples=3)
    assert res["prob_var"].shape == (B, T)
    