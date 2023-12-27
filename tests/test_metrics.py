import torch as t
from dishonesty.utils import calc_soft_kl_div


def test_soft_kl_div():
    # Simulate [batch, pos, d_vocab]
    logits1 = t.randn(2, 3, 1000)
    logits2 = t.randn(2, 3, 1000)

    # Test that soft KLD is equal to vanilla KLD when temperature=1.0
    soft_kld_t1 = calc_soft_kl_div(logits1, logits2, temperature=1.0)
    vanilla_kld = t.nn.functional.kl_div(
        logits2.log_softmax(dim=-1),
        logits1.log_softmax(dim=-1),
        log_target=True,
        reduction="none",
    ).sum(dim=-1)

    assert t.allclose(soft_kld_t1, vanilla_kld)

    # Test that soft KLD is equal to logprob diff when temperature=0.0
    soft_kld_t0 = calc_soft_kl_div(logits1, logits2, temperature=0.0)
    argmax_indices = logits1.argmax(dim=-1, keepdim=True)
    logprob_diffs = logits1.log_softmax(dim=-1) - logits2.log_softmax(dim=-1)
    argmax_logprob_diffs = logprob_diffs.gather(dim=-1, index=argmax_indices).squeeze(-1)

    assert t.allclose(soft_kld_t0, argmax_logprob_diffs)
