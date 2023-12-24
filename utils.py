import torch as t

from functools import partial


def _cache_hook(module, input, output, name=None, cache=None):
    if cache is None:
        raise Exception("Cache is not initialized.")
    cache[name] = output

def _cache_hook_tuple(module, input, output, name=None, cache=None):
    if cache is None:
        raise Exception("Cache is not initialized.")
    cache[name] = output[0]

def _cache_hook_final_rscale(module, input, output, cache=None):
    if cache is None:
        raise Exception("Cache is not initialized.")
    hidden_states = input[0]
    hidden_states = hidden_states.to(t.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    cache["final_rscale"] = t.rsqrt(variance + module.variance_epsilon)

def _get_add_hook_args(hook_name, cache):
    if hook_name == "embed_tokens":
        module_name = "model.embed_tokens"
        hook_fnc = partial(_cache_hook, name=hook_name, cache=cache)
    elif "resid_post" in hook_name:
        layer = int(hook_name.split("_")[-1])
        module_name = f"model.layers.{layer}"
        hook_fnc = partial(_cache_hook_tuple, name=hook_name, cache=cache)
    elif "attn_out" in hook_name:
        layer = int(hook_name.split("_")[-1])
        module_name = f"model.layers.{layer}.self_attn"
        hook_fnc = partial(_cache_hook_tuple, name=hook_name, cache=cache)
    elif "mlp_out" in hook_name:
        layer = int(hook_name.split("_")[-1])
        module_name = f"model.layers.{layer}.mlp"
        hook_fnc = partial(_cache_hook, name=hook_name, cache=cache)
    elif hook_name == "final_rscale":
        module_name = "model.norm"
        hook_fnc = partial(_cache_hook_final_rscale, cache=cache)
    elif hook_name == "final_norm_out":
        module_name = "model.norm"
        hook_fnc = partial(_cache_hook, name=hook_name, cache=cache)
    else:
        raise Exception(f"Unsupported hook point: {hook_name}")

    # Return args to pass to self.add_hooks
    return module_name, hook_fnc


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
        print(f"Cache size: {num_bytes / 1e6 :.1d} MB")


# See: https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py
class HookedMistral:
    def __init__(self, hf_model, tokenizer):
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.device = hf_model.device

    def __repr__(self):
        return (
            "Supported hook points:\n"
            "embed_tokens\n"
            "attn_out_{0..31}\n"
            "mlp_out_{0..31}\n"
            "resid_post_{0..31}\n"
            "final_rscale\n"
            "final_norm_out"
        )

    def __call__(self, prompts, attention_mask=None):
        if isinstance(prompts, str) or (
            isinstance(prompts, list) and isinstance(prompts[0], str)
        ):
            tokens, attention_mask = self.to_tokens(prompts, return_mask=True)
        else:
            tokens = prompts
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
                mask = t.ones_like(tokens).to(device)
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

    def get_resid_post_names(self, layers_filter=None):
        if isinstance(layers_filter, int):
            layers_filter = [layers_filter]
        if layers_filter is None:
            layers_filter = range(self.hf_model.config.num_hidden_layers)
        return [f"model.layers.{l}" for l in layers_filter]

    def get_component_names(self, layers_filter=None):
        if isinstance(layers_filter, int):
            layers_filter = [layers_filter]
        if layers_filter is None:
            layers_filter = range(self.hf_model.config.num_hidden_layers)
        names = []
        for l in layers_filter:
            names.append(f"model.layers.{l}.self_attn")
            names.append(f"model.layers.{l}.mlp")
        return names

    def get_module(self, name_to_get):
        for name, module in self.hf_model.named_modules():
            if name == name_to_get:
                return module
        raise Exception(f"{name_to_get} not found.")

    def add_hook(self, name, hook_fnc):
        module = self.get_module(name)
        handle = module.register_forward_hook(hook_fnc)
        return handle

    def reset_hooks(self):
        for module in self.hf_model.modules():
            module._forward_hooks.clear()
            module._forward_pre_hooks.clear()

    def run_with_cache(self, prompts, names, attention_mask=None):
        # Init cache and add hooks
        cache = MistralCache()
        handles = []
        for name in names:
            handle = self.add_hook(*_get_add_hook_args(name, cache))
            handles.append(handle)

        # Get tokens and run a forward pass
        with t.no_grad():
            logits = self.__call__(prompts, attention_mask=attention_mask)

        # Remove cache hooks
        for handle in handles:
            handle.remove()

        return logits, cache
