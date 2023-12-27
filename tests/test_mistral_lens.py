# Handy snippet to get repo root from anywhere in the repo
import sys
from subprocess import check_output

ROOT = check_output("git rev-parse --show-toplevel", shell=True).decode("utf-8").strip()
if ROOT not in sys.path:
    sys.path.append(ROOT)

# Actual imports
import torch as t
from mistral_lens import load_model

t.set_grad_enabled(False)
device = "cuda" if t.cuda.is_available() else "cpu"
model = load_model()


def test_attention_mask():
    # Check to see if HookedMistral uses attention masking like transformers
    prompts = [
        "This sentence will be left-padded.",
        "This somewhat longer sentence won't be left-padded.",
    ]
    tokens, mask = model.tokenizer(prompts, padding=True, return_tensors="pt").values()
    tokens, mask = tokens.to(device), mask.to(device)

    logits_hf = model.hf_model(input_ids=tokens, attention_mask=mask).logits
    logits_fwd = model(prompts)
    logits_cache, _ = model.run_with_cache(prompts, ["embed_tokens"])

    assert t.allclose(logits_hf.cpu(), logits_fwd.cpu())
    assert t.allclose(logits_hf.cpu(), logits_cache.cpu())


def test_residual_decomposition():
    # Check if decomposing then accumulating the residual stream will match the last resid_post
    names = ["embed_tokens", "resid_post_31"]
    for i in range(32):
        names.append(f"attn_out_{i}")
        names.append(f"mlp_out_{i}")

    logits, cache = model.run_with_cache("Will the tensors match?", names)
    resids_decomp = cache.get_decomposed_resid_stack()
    resids_accum = resids_decomp.cumsum(dim=0)

    assert t.allclose(resids_accum[-1], cache["resid_post_31"])


def test_rms_norm():
    # Check if manually scaling resid_post_31 will match final_norm_out
    def apply_rms_norm(resid, rscale, weight):
        input_dtype = resid.dtype
        scaled_resid = rscale * resid.to(rscale.dtype) * weight.to(rscale.dtype)
        return scaled_resid.to(input_dtype)

    names = ["final_rscale", "resid_post_31", "final_norm_out"]
    logits, cache = model.run_with_cache("Will the tensors match?", names)
    manual_norm_out = apply_rms_norm(
        cache["resid_post_31"],
        cache["final_rscale"],
        model.hf_model.model.norm.weight,
    )

    assert t.allclose(
        manual_norm_out,
        cache["final_norm_out"],
        atol=1e-6,
        rtol=1e-3,
    )


def test_head_outputs():
    # Check if individual head outputs sum to the attn block output
    names = []
    for i in range(32):
        names += [f"head_out_{i}", f"attn_out_{i}"]

    logits, cache = model.run_with_cache("Will the tensors match?", names)

    for i in range(32):
        assert t.allclose(
            cache[f"head_out_{i}"].sum(dim=2),
            cache[f"attn_out_{i}"],
            atol=1e-4,
            rtol=1e-3,
        )
