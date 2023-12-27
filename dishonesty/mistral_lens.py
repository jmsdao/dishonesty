import torch as t
import torch
import torch.nn as nn
import math
import einops

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if t.cuda.is_available() else "cpu"


def load_model(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    torch_dtype=t.float16,
    device_map=device,
    cache_dir="/workspace/cache",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, padding_side="left")
    tokenizer.pad_token_id = 1
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        cache_dir=cache_dir,
    )
    return HookedMistral(hf_model, tokenizer)


class MistralCache:
    def __init__(self):
        self.store = {}

    def __repr__(self):
        string = ""
        for key, tensor in self.store.items():
            string += f'["{key}"] : {tensor.shape}\n'
        return string.rstrip()

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def keys(self):
        return self.store.keys()
    
    def values(self):
        return self.store.values()

    def items(self):
        return self.store.items()

    def get_decomposed_resid_stack(self, return_labels=False):
        """Take the cumsum along the comp dimension if you wanted accumulated resid."""
        stack_names = ["embed_tokens"]
        for i in range(32):
            stack_names.append(f"attn_out_{i}")
            stack_names.append(f"mlp_out_{i}")

        resid_stack = t.stack([self.store[name] for name in stack_names], dim=0)  # [comp, batch, pos, d_model]

        if return_labels:
            return resid_stack, stack_names
        else:
            return resid_stack
      
    def size(self):
        num_bytes = 0
        for tensor in self.store.values():
            num_bytes += tensor.numel() * tensor.dtype.itemsize
        print(f"Cache size: {num_bytes / 1e6 :.1f} MB")


# See: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py
class HookedMistral:
    def __init__(self, hf_model, tokenizer):
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.device = hf_model.device

    def __repr__(self):
        return (
            "Supported caches:\n"
            "embed_tokens\n"
            "attn_out_{0..31}\n"
            "head_out_{0..31}\n"
            "mlp_out_{0..31}\n"
            "resid_post_{0..31}\n"
            "final_rscale\n"
            "final_norm_out\n"
            "\n"
            "Supported hook points:\n"
            "embed_tokens\n"
            "attn_out_{0..31}\n"
            "mlp_out_{0..31}\n"
            "resid_post_{0..31}\n"
        )

    def __call__(self, prompts):
        tokens, attention_mask = self.to_tokens(prompts, return_mask=True)
        with t.no_grad():
            out = self.hf_model(input_ids=tokens, attention_mask=attention_mask)
        return out.logits

    def to_tokens(self, prompts, device=None, return_mask=False):
        if device is None:
            device = self.device
        # Convert string to list
        if isinstance(prompts, str):
            prompts = [prompts]
        # If the input is a list of strings
        if isinstance(prompts, list) and isinstance(prompts[0], str):
            tokens, attention_mask = self.tokenizer.batch_encode_plus(
                prompts,
                padding=True,
                return_tensors="pt",
            ).values()
            tokens = tokens.to(device)
            attention_mask = attention_mask.to(device)
            if return_mask:
                return tokens, attention_mask
            return tokens
        # If the input is a tensor
        elif isinstance(prompts, t.Tensor):
            if prompts.dim() == 1:
                prompts = prompts.unsqueeze(0)
            tokens = prompts.to(device)
            if return_mask:
                # Infer mask from tokens
                mask_counts = t.stack([t.bincount(tid, minlength=tokens.shape[-1])[1] for tid in tokens]) - 1
                mask = t.ones_like(tokens)
                for i, mc in enumerate(mask_counts):
                    mask[i, 0:mc] = 0
                return tokens, mask
            return tokens
        else:
            raise ValueError("Invalid input type")

    def to_string(self, tokens):
        if isinstance(tokens, t.Tensor) and tokens.dim() == 1:
            strings = self.tokenizer.decode(tokens)
        else:
            strings = self.tokenizer.batch_decode(tokens)
        return strings

    def to_string_tokenized(self, tokens):
        if isinstance(tokens, str) or (
            isinstance(tokens, list) and isinstance(tokens[0], str)
        ):
            tokens = self.to_tokens(tokens)[0]
        if isinstance(tokens, t.Tensor) and tokens.dim() == 1:
            return self.tokenizer.batch_decode(tokens)
        return [self.tokenizer.batch_decode(t) for t in tokens]

    def get_module(self, name):
        module_map = {
            module_name: module
            for module_name, module in self.hf_model.named_modules()
        }
        if name not in module_map:
            name = hook_name_to_module_name(name)
        if name not in module_map:
            raise Exception(f"{name} not found.")
        return module_map[name]

    def add_hook(self, name, hook_fnc):
        module = self.get_module(name)
        handle = module.register_forward_hook(hook_fnc)
        return handle

    def reset_hooks(self):
        for module in self.hf_model.modules():
            module._forward_hooks.clear()
            module._forward_pre_hooks.clear()

    def run_with_cache(self, prompts, names):
        # Get attention mask
        _, attention_mask = self.to_tokens(prompts, return_mask=True)

        # Init cache and add hooks
        cache = MistralCache()
        handles = []
        for name in names:
            handle = self.add_hook(*_get_cache_hook_args(self, name, cache, attention_mask))
            handles.append(handle)

        # Get tokens and run a forward pass
        with t.no_grad():
            logits = self.__call__(prompts)

        # Remove cache hooks
        for handle in handles:
            handle.remove()

        return logits, cache


# =========================== HELPERS FOR HOOKS  =========================== #
def hook_name_to_module_name(hook_name):
    if hook_name == "embed_tokens":
        module_name = "model.embed_tokens"
    elif "attn_out" in hook_name:
        layer = int(hook_name.split("_")[-1])
        module_name = f"model.layers.{layer}.self_attn"
    elif "mlp_out" in hook_name:
        layer = int(hook_name.split("_")[-1])
        module_name = f"model.layers.{layer}.mlp"
    elif "resid_post" in hook_name:
        layer = int(hook_name.split("_")[-1])
        module_name = f"model.layers.{layer}"
    else:
        raise Exception(f"Unsupported hook point: {hook_name}")

    return module_name

# ==================== HELPERS FOR CACHING ACTIVATIONS ==================== #
def _cache_hook(module, input, output, hook_name=None, cache=None):
    if cache is None:
        raise Exception("Cache is not initialized.")
    cache[hook_name] = output


def _cache_hook_tuple(module, input, output, hook_name=None, cache=None):
    if cache is None:
        raise Exception("Cache is not initialized.")
    cache[hook_name] = output[0]


def _cache_hook_head(
    module,
    input,
    output,
    hook_name=None,
    cache=None,
    attn_module=None,
    attention_mask=None
):
    if cache is None:
        raise Exception("Cache is not initialized.")
    if attn_module is None:
        raise Exception("Attention module is not initialized.")
    
    if attention_mask is not None:
        position_ids = attention_mask.cumsum(dim=-1) - 1
    else:
        position_ids = None

    attention_mask_4d = _prepare_4d_causal_attention_mask(
        attention_mask,
        (output.shape[0], output.shape[1]),
        output,
        0,
    )

    cache[hook_name] = compute_output_per_head(
        attn_module,
        output,
        attention_mask_4d,
        position_ids,
    )


def _cache_hook_final_rscale(module, input, output, cache=None):
    if cache is None:
        raise Exception("Cache is not initialized.")
    hidden_states = input[0]
    hidden_states = hidden_states.to(t.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    cache["final_rscale"] = t.rsqrt(variance + module.variance_epsilon)


def _get_cache_hook_args(hooked_model, hook_name, cache, attention_mask):
    if hook_name == "embed_tokens":
        module_name = "model.embed_tokens"
        hook_fnc = partial(_cache_hook, hook_name=hook_name, cache=cache)
    elif "resid_post" in hook_name:
        layer = int(hook_name.split("_")[-1])
        module_name = f"model.layers.{layer}"
        hook_fnc = partial(_cache_hook_tuple, hook_name=hook_name, cache=cache)
    elif "attn_out" in hook_name:
        layer = int(hook_name.split("_")[-1])
        module_name = f"model.layers.{layer}.self_attn"
        hook_fnc = partial(_cache_hook_tuple, hook_name=hook_name, cache=cache)
    elif "head_out" in hook_name:
        layer = int(hook_name.split("_")[-1])
        module_name = f"model.layers.{layer}.input_layernorm"
        attn_module = hooked_model.get_module(f"model.layers.{layer}.self_attn")
        hook_fnc = partial(
            _cache_hook_head,
            hook_name=hook_name,
            cache=cache,
            attn_module=attn_module,
            attention_mask=attention_mask,
        )
    elif "mlp_out" in hook_name:
        layer = int(hook_name.split("_")[-1])
        module_name = f"model.layers.{layer}.mlp"
        hook_fnc = partial(_cache_hook, hook_name=hook_name, cache=cache)
    elif hook_name == "final_rscale":
        module_name = "model.norm"
        hook_fnc = partial(_cache_hook_final_rscale, cache=cache)
    elif hook_name == "final_norm_out":
        module_name = "model.norm"
        hook_fnc = partial(_cache_hook, hook_name=hook_name, cache=cache)
    else:
        raise Exception(f"Unsupported hook point: {hook_name}")

    # Return args to pass to self.add_hooks
    return module_name, hook_fnc


# ================= HELPERS FOR COMPUTING PER HEAD OUTPUTS ================= #
@dataclass
class AttentionMaskConverter:
    """
    A utility attention mask class that allows one to:
        - Create a causal 4d mask
        - Create a causal 4d mask with slided window
        - Convert a 2d attention mask (batch_size, query_length) to a 4d attention mask (batch_size, 1, query_length,
          key_value_length) that can be multiplied with attention scores

    Examples:

    ```python
    >>> import torch
    >>> from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    >>> converter = AttentionMaskConverter(True)
    >>> converter.to_4d(torch.tensor([[0, 0, 0, 1, 1]]), 5, key_value_length=5, dtype=torch.float32)
    tensor([[[[-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00,  0.0000e+00]]]])
    ```

    Parameters:
        is_causal (`bool`):
            Whether the attention mask should be a uni-directional (causal) or bi-directional mask.

        sliding_window (`int`, *optional*):
            Optionally, the sliding window masks can be created if `sliding_window` is defined to a positive integer.
    """

    is_causal: bool
    sliding_window: int

    def __init__(self, is_causal: bool, sliding_window: Optional[int] = None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window

        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: torch.dtype,
        device: Union[torch.device, "str"] = "cpu",
    ) -> Optional[torch.Tensor]:
        """
        Creates a causal 4D mask of (bsz, head_dim=1, query_length, key_value_length) shape and adds large negative
        bias to upper right hand triangular matrix (causal mask).
        """
        if not self.is_causal:
            raise ValueError(f"Please use `to_causal_4d` only if {self.__class__} has `is_causal` set to True.")

        # If shape is not cached, create a new causal mask and cache it
        input_shape = (batch_size, query_length)
        past_key_values_length = key_value_length - query_length

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if input_shape[-1] > 1 or self.sliding_window is not None:
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )

        return causal_4d_mask

    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
        key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
        causal, a causal mask will be added.
        """
        input_shape = (attention_mask_2d.shape[0], query_length)

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        causal_4d_mask = None
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )

            past_key_values_length = key_value_length - query_length
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=attention_mask_2d.device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
            )
        elif self.sliding_window is not None:
            raise NotImplementedError("Sliding window is currently only implemented for causal masking")

        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = self._expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1]).to(
            attention_mask_2d.device
        )
        if causal_4d_mask is not None:
            expanded_attn_mask = causal_4d_mask.masked_fill(expanded_attn_mask.bool(), torch.finfo(dtype).min)

        # expanded_attn_mask + causal_4d_mask can cause some overflow
        expanded_4d_mask = expanded_attn_mask

        return expanded_4d_mask

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window: Optional[int] = None,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

        # add lower triangular sliding window mask if necessary
        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window + 1

            context_mask = 1 - torch.triu(torch.ones_like(mask, dtype=torch.int), diagonal=diagonal)
            mask.masked_fill_(context_mask.bool(), torch.finfo(dtype).min)

        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    @staticmethod
    def _unmask_unattended(
        expanded_mask: torch.Tensor, attention_mask: torch.Tensor, unmasked_value: Union[bool, float]
    ):
        # fmt: off
        """
        Attend to all tokens in masked rows from the expanded attention mask, for example the relevant first rows when
        using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        Details: https://github.com/pytorch/pytorch/issues/110213

        `expanded_mask` is [bsz, num_masks, tgt_seq_len, src_seq_len] or [bsz, tgt_seq_len, src_seq_len].
        `attention_mask` is [bsz, src_seq_len].

        The dimension num_masks of `expanded_mask` is most often 1, but it can also be the number of heads in the case of alibi attention bias.

        For example, if `attention_mask` is
        ```
        [[0, 0, 1],
         [1, 1, 1],
         [0, 1, 1]]
        ```
        and `expanded_mask` is (e.g. here left-padding case)
        ```
        [[[[0, 0, 0],
           [0, 0, 0],
           [0, 0, 1]]],
         [[[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]]],
         [[[0, 0, 0],
           [0, 1, 0],
           [0, 1, 1]]]]
        ```
        then the modified `expanded_mask` will be
        ```
        [[[[1, 1, 1],   <-- modified
           [1, 1, 1],   <-- modified
           [0, 0, 1]]],
         [[[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]]],
         [[[1, 1, 1],   <-- modified
           [0, 1, 0],
           [0, 1, 1]]]]
        ```
        """
        # fmt: on

        # Get the index of the first non-zero value for every sample in the batch.
        # In the above example, indices = [[2], [0], [1]]]
        tmp = torch.arange(attention_mask.shape[1], 0, -1)
        indices = torch.argmax(attention_mask.cpu() * tmp, 1, keepdim=True)

        # Find the batch indexes that have unattended tokens on the leftmost side (e.g. [0, 0, 1, 1, 1]), for which the first rows of the
        # expanded mask will be completely unattended.
        left_masked_rows = torch.where(indices > 0)[0]

        if left_masked_rows.shape[0] == 0:
            return expanded_mask
        indices = indices[left_masked_rows]

        max_len = torch.max(indices)
        range_tensor = torch.arange(max_len).unsqueeze(0)
        range_tensor = range_tensor.repeat(indices.size(0), 1)

        # Avoid unmasking tokens at relevant target positions (on the row axis), by rather unmasking possibly several times the first row that should always be unmasked as we filtered out the batch above.
        range_tensor[range_tensor >= indices] = 0

        # TODO: we may drop support for 3D attention mask as the refactor from Patrick maybe dropped this case
        if expanded_mask.dim() == 4:
            num_masks = expanded_mask.shape[1]
            if num_masks == 1:
                # Broadcast [left_masked_rows, 1], [left_masked_rows, max_len]
                mask_slice = (left_masked_rows[:, None], 0, range_tensor)
            else:
                # Broadcast [left_masked_rows, 1, 1], [1, num_masks, 1], [left_masked_rows, 1, max_len]
                mask_slice = (
                    left_masked_rows[:, None, None],
                    torch.arange(num_masks)[None, :, None],
                    range_tensor[:, None, :],
                )
        else:
            # Broadcast [left_masked_rows, 1], [left_masked_rows, max_len]
            mask_slice = (left_masked_rows[:, None], range_tensor)

        expanded_mask[mask_slice] = unmasked_value

        return expanded_mask


def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`torch.Tensor`):
            The embedded inputs as a torch Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None and len(attention_mask.shape) == 2:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length=key_value_length, dtype=inputs_embeds.dtype
        )
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            # if the 4D mask has correct shape - invert it and fill with negative infinity
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
            )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    return attention_mask


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def compute_output_per_head(
    attn_module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = attn_module.q_proj(hidden_states)
    key_states = attn_module.k_proj(hidden_states)
    value_states = attn_module.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, attn_module.num_heads, attn_module.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    cos, sin = attn_module.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, attn_module.num_key_value_groups)
    value_states = repeat_kv(value_states, attn_module.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn_module.head_dim)

    if attn_weights.size() != (bsz, attn_module.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, attn_module.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)  # [batch, head, pos, d_head]

    return einops.einsum(
        attn_output,
        attn_module.o_proj.weight.T.reshape(32, 128, 4096),
        "batch head pos d_head, head d_head d_model -> batch pos head d_model"
    )
