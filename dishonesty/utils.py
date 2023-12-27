def calc_soft_kl_div(target_logits, input_logits, temperature=1.0, eps=1e-6):
    # When temperature is 0.0, calculate using argmax for numerical stability.
    if temperature == 0.0:
        argmax_indices = target_logits.argmax(dim=-1, keepdim=True)
        target_logprobs = target_logits.log_softmax(dim=-1)
        input_logprobs = input_logits.log_softmax(dim=-1)
        target_logprobs_at_argmax = target_logprobs.gather(dim=-1, index=argmax_indices)
        input_logprobs_at_argmax = input_logprobs.gather(dim=-1, index=argmax_indices)
        return (target_logprobs_at_argmax - input_logprobs_at_argmax).squeeze(-1)

    if temperature <= 0.01:
        print("Warning: be aware of numerical instability as temperature gets very small.")

    probs = (target_logits / (temperature + eps)).softmax(dim=-1)
    logprob_diffs = target_logits.log_softmax(dim=-1) - input_logits.log_softmax(dim=-1)
    return (probs * logprob_diffs).sum(dim=-1)
