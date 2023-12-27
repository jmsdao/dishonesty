def calc_soft_kl_div(target_logits, input_logits, temperature=1.0, eps=1e-6):
    probs = (target_logits / (temperature + eps)).softmax(dim=-1)
    logprob_diffs = target_logits.log_softmax(dim=-1) - input_logits.log_softmax(dim=-1)
    return (probs * logprob_diffs).sum(dim=-1)
