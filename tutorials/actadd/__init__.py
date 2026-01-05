"""
Activation Addition (ActAdd) Implementation using pyvene

This module provides a pyvene-based implementation of the ActAdd technique
for steering language models, matching the API from the original paper.

Reference: "Activation Addition: Steering Language Models Without Optimization"

Usage:
    import actadd
    
    # Get a steering vector
    steering_vec = actadd.get_diff_vector(model, tokenizer, "Love", "Hate", layer=6)
    
    # Apply during generation
    outputs = actadd.generate_with_steering(
        model, tokenizer, "I hate you because",
        steering_vec, layer=6, coeff=5.0
    )
"""

from typing import List, Optional, Tuple, Dict, Union, Any
from dataclasses import dataclass
from contextlib import contextmanager
import torch
import torch.nn as nn
import pyvene as pv


# ============================================================================
# Core Functions
# ============================================================================

def get_blocks(model: nn.Module) -> nn.ModuleList:
    """
    Get the ModuleList containing the transformer blocks from a model.
    
    This automatically detects the transformer blocks by finding the ModuleList
    that contains >50% of the model's parameters.
    """
    def numel_(mod):
        return sum(p.numel() for p in mod.parameters())
    
    model_numel = numel_(model)
    candidates = [
        mod
        for mod in model.modules()
        if isinstance(mod, nn.ModuleList) and numel_(mod) > 0.5 * model_numel
    ]
    assert len(candidates) == 1, f'Found {len(candidates)} ModuleLists with >50% of model params.'
    return candidates[0]


def _device(model: nn.Module) -> torch.device:
    """Get the device of the model."""
    return next(model.parameters()).device


def get_vectors(
    model: nn.Module, 
    tokenizer, 
    prompts: List[str], 
    layer: int
) -> torch.Tensor:
    """
    Get the activations of the prompts at the specified layer.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        prompts: List of prompts to get activations for
        layer: Which layer to extract activations from
        
    Returns:
        Tensor of shape (batch, seq_len, hidden_size) with activations
    """
    # Configure pyvene to collect activations
    config = pv.IntervenableConfig(
        representations=[
            pv.RepresentationConfig(
                layer=layer,
                component="block_output",
                intervention=pv.CollectIntervention()
            )
        ]
    )
    
    intervenable = pv.IntervenableModel(config, model)
    device = _device(model)
    
    # Tokenize with padding
    inputs = tokenizer(prompts, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get the sequence length for unit_locations
    seq_len = inputs['input_ids'].shape[1]
    
    # Forward pass to collect activations
    (_, collected), _ = intervenable(
        base=inputs,
        unit_locations={"base": list(range(seq_len))}
    )
    
    return collected[0]  # Shape: (batch, seq_len, hidden_size)


def get_diff_vector(
    model: nn.Module, 
    tokenizer, 
    prompt_add: str, 
    prompt_sub: str, 
    layer: int
) -> torch.Tensor:
    """
    Get the difference vector between two prompts at the specified layer.
    
    This is the core of ActAdd: steering_vector = activations(prompt_add) - activations(prompt_sub)
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        prompt_add: The positive prompt (steer toward this)
        prompt_sub: The negative prompt (steer away from this)
        layer: Which layer to extract the vector from
        
    Returns:
        Steering vector tensor of shape (1, seq_len, hidden_size)
    """
    stream = get_vectors(model, tokenizer, [prompt_add, prompt_sub], layer)
    # Return difference, adding batch dimension
    return (stream[0] - stream[1]).unsqueeze(0)


# ============================================================================
# ActivationAddition Dataclass (matching original API)
# ============================================================================

@dataclass
class ActivationAddition:
    """
    Represents an activation addition intervention.
    
    Add coeff * act to the residual stream at the specified layer.
    
    Attributes:
        prompt: The prompt used to generate this activation (for logging)
        coeff: Scaling coefficient for the activation
        layer: Which layer to apply the addition
        act: The activation tensor to add (shape: batch, seq_len, hidden_size)
    """
    prompt: str
    coeff: float
    layer: int
    act: torch.Tensor
    
    def __post_init__(self):
        assert len(self.act.shape) == 3, \
            f"act must be (batch, seq_len, dim) but got shape {self.act.shape}"
    
    def __repr__(self):
        return (f"ActivationAddition(coeff={self.coeff}, layer={self.layer}, "
                f"prompt='{self.prompt}', act.shape={self.act.shape})")


def get_x_vector(
    prompt1: str,
    prompt2: str,
    coeff: float,
    act_name: int,  # layer number
    model: nn.Module,
    tokenizer=None,
    pad_method: Optional[str] = None,
    custom_pad_id: Optional[int] = None,
) -> List[ActivationAddition]:
    """
    Get activation additions for a pair of contrasting prompts.
    
    This matches the original ActAdd API. Returns two ActivationAddition objects:
    one for prompt1 with positive coeff, one for prompt2 with negative coeff.
    
    Args:
        prompt1: First prompt (positive direction)
        prompt2: Second prompt (negative direction)
        coeff: Scaling coefficient
        act_name: Layer number for intervention
        model: The transformer model
        tokenizer: The tokenizer (if None, uses model.tokenizer)
        pad_method: Padding method (currently only "tokens_right" supported)
        custom_pad_id: Custom padding token ID
        
    Returns:
        List of two ActivationAddition objects
    """
    if tokenizer is None:
        assert hasattr(model, 'tokenizer'), "Model must have model.tokenizer or pass tokenizer"
        tokenizer = model.tokenizer
        
    if pad_method is not None and pad_method != "tokens_right":
        raise NotImplementedError('pad_method != "tokens_right" is not implemented')
    
    original_pad_id = tokenizer.pad_token_id
    if custom_pad_id is not None:
        tokenizer.pad_token_id = custom_pad_id
    
    try:
        act = get_vectors(model, tokenizer, [prompt1, prompt2], act_name)
    finally:
        tokenizer.pad_token_id = original_pad_id
    
    return [
        ActivationAddition(
            prompt=prompt1, 
            coeff=coeff, 
            layer=act_name, 
            act=act[0].unsqueeze(0)
        ),
        ActivationAddition(
            prompt=prompt2, 
            coeff=-coeff, 
            layer=act_name, 
            act=act[1].unsqueeze(0)
        ),
    ]


# ============================================================================
# Steering with pyvene
# ============================================================================

def create_steering_config(
    additions: List[ActivationAddition],
    model: nn.Module
) -> pv.IntervenableConfig:
    """
    Create a pyvene IntervenableConfig from ActivationAddition objects.
    
    Args:
        additions: List of ActivationAddition objects to apply
        model: The model (used to determine device)
        
    Returns:
        IntervenableConfig for the steering intervention
    """
    device = _device(model)
    
    representations = []
    for add in additions:
        # Scale the activation by coefficient
        scaled_act = (add.coeff * add.act).to(device)
        
        representations.append(
            pv.RepresentationConfig(
                layer=add.layer,
                component="block_output",
                intervention=pv.AdditionIntervention(
                    source_representation=scaled_act
                )
            )
        )
    
    return pv.IntervenableConfig(representations=representations)


def create_steered_model(
    model: nn.Module,
    additions: List[ActivationAddition]
) -> pv.IntervenableModel:
    """
    Create an IntervenableModel with steering applied.
    
    Args:
        model: The base model
        additions: List of ActivationAddition objects
        
    Returns:
        IntervenableModel configured for steering
    """
    config = create_steering_config(additions, model)
    intervenable = pv.IntervenableModel(config, model)
    intervenable.set_device(_device(model))
    return intervenable


def generate_with_steering(
    model: nn.Module,
    tokenizer,
    prompt: str,
    additions: List[ActivationAddition],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.3,
    do_sample: bool = True,
    **kwargs
) -> str:
    """
    Generate text with steering vectors applied.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        prompt: Input prompt
        additions: List of ActivationAddition objects
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to sample (True) or use greedy decoding (False)
        **kwargs: Additional generation arguments
        
    Returns:
        Generated text string
    """
    intervenable = create_steered_model(model, additions)
    device = _device(model)
    
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    seq_len = inputs['input_ids'].shape[1]
    
    _, output = intervenable.generate(
        base=inputs,
        unit_locations={"base": list(range(seq_len))},
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        **kwargs
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)


# ============================================================================
# Comparison Utilities
# ============================================================================

def generate_comparison(
    model: nn.Module,
    tokenizer,
    prompt: str,
    additions: List[ActivationAddition],
    num_samples: int = 3,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.3,
    seed: Optional[int] = None,
    **kwargs
) -> Dict[str, List[str]]:
    """
    Generate completions with and without steering for comparison.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        prompt: Input prompt
        additions: List of ActivationAddition objects
        num_samples: Number of samples to generate
        max_new_tokens: Maximum tokens per sample
        temperature: Sampling temperature
        top_p: Top-p sampling
        seed: Random seed for reproducibility
        **kwargs: Additional generation arguments
        
    Returns:
        Dict with 'unsteered' and 'steered' completion lists
    """
    device = _device(model)
    
    if seed is not None:
        torch.manual_seed(seed)
    
    results = {
        'prompt': prompt,
        'unsteered': [],
        'steered': []
    }
    
    # Create inputs batch
    prompts = [prompt] * num_samples
    inputs = tokenizer(prompts, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        **kwargs
    )
    
    # Generate unsteered
    with torch.no_grad():
        unsteered_ids = model.generate(**inputs, **gen_kwargs)
    
    for ids in unsteered_ids:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        # Remove prompt from completion
        completion = text[len(prompt):] if text.startswith(prompt) else text
        results['unsteered'].append(completion)
    
    # Generate steered
    intervenable = create_steered_model(model, additions)
    seq_len = inputs['input_ids'].shape[1]
    
    _, steered_ids = intervenable.generate(
        base=inputs,
        unit_locations={"base": list(range(seq_len))},
        **gen_kwargs
    )
    
    for ids in steered_ids:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        completion = text[len(prompt):] if text.startswith(prompt) else text
        results['steered'].append(completion)
    
    return results


def print_comparison(
    results: Dict[str, List[str]],
    unsteered_title: str = "Unsteered",
    steered_title: str = "Steered"
) -> None:
    """
    Pretty-print comparison results.
    
    Args:
        results: Dict from generate_comparison
        unsteered_title: Title for unsteered column
        steered_title: Title for steered column
    """
    prompt = results['prompt']
    unsteered = results['unsteered']
    steered = results['steered']
    
    # ANSI bold
    bold = lambda s: f"\033[1m{s}\033[0m"
    
    # Calculate column width
    width = 60
    separator = "+" + "-" * (width + 2) + "+" + "-" * (width + 2) + "+"
    
    print(separator)
    print(f"| {bold(unsteered_title):^{width+9}} | {bold(steered_title):^{width+9}} |")
    print(separator)
    
    for u, s in zip(unsteered, steered):
        # Truncate long completions
        u_display = (u[:width-3] + "...") if len(u) > width else u
        s_display = (s[:width-3] + "...") if len(s) > width else s
        
        # Format with prompt highlighted
        u_line = f"{bold(prompt)}{u_display}"
        s_line = f"{bold(prompt)}{s_display}"
        
        print(f"| {u_display:<{width}} | {s_display:<{width}} |")
        print(separator)


def get_n_comparisons(
    prompts: List[str],
    model: nn.Module,
    additions: List[ActivationAddition],
    tokenizer=None,
    **sampling_kwargs
) -> "pd.DataFrame":
    """
    Generate comparisons and return as a DataFrame.
    
    Matches the original ActAdd API for compatibility.
    
    Args:
        prompts: List of prompts
        model: The model
        additions: List of ActivationAddition objects
        tokenizer: The tokenizer (if None, uses model.tokenizer)
        **sampling_kwargs: Generation parameters
        
    Returns:
        DataFrame with columns: prompts, completions, is_modified
    """
    import pandas as pd
    
    if tokenizer is None:
        tokenizer = model.tokenizer
    
    device = _device(model)
    inputs = tokenizer(prompts, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Handle sampling kwargs
    gen_kwargs = {}
    if 'tokens_to_generate' in sampling_kwargs:
        gen_kwargs['max_new_tokens'] = sampling_kwargs['tokens_to_generate']
    if 'temperature' in sampling_kwargs:
        gen_kwargs['temperature'] = sampling_kwargs['temperature']
    if 'top_p' in sampling_kwargs:
        gen_kwargs['top_p'] = sampling_kwargs['top_p']
    if 'seed' in sampling_kwargs:
        torch.manual_seed(sampling_kwargs['seed'])
    
    gen_kwargs['do_sample'] = True
    gen_kwargs['pad_token_id'] = tokenizer.eos_token_id
    
    def to_df(tokens, modified):
        completions = [tokenizer.decode(t.tolist(), skip_special_tokens=True) for t in tokens]
        trimmed = [c[len(p):] for p, c in zip(prompts, completions)]
        return pd.DataFrame({
            'prompts': prompts,
            'completions': trimmed,
            'is_modified': modified,
        })
    
    # Generate unmodified
    with torch.no_grad():
        nom_tokens = model.generate(**inputs, **gen_kwargs)
    
    # Generate modified
    intervenable = create_steered_model(model, additions)
    seq_len = inputs['input_ids'].shape[1]
    
    _, mod_tokens = intervenable.generate(
        base=inputs,
        unit_locations={"base": list(range(seq_len))},
        **gen_kwargs
    )
    
    nom_df = to_df(nom_tokens, modified=False)
    mod_df = to_df(mod_tokens, modified=True)
    
    return pd.concat([nom_df, mod_df], ignore_index=True)


def print_n_comparisons(
    prompt: str,
    model: nn.Module,
    num_comparisons: int = 5,
    activation_additions: Optional[List[ActivationAddition]] = None,
    tokenizer=None,
    **kwargs
) -> None:
    """
    Print comparison table matching the original ActAdd API.
    
    Args:
        prompt: The input prompt
        model: The model
        num_comparisons: Number of samples
        activation_additions: List of ActivationAddition objects
        tokenizer: The tokenizer
        **kwargs: Sampling parameters
    """
    if tokenizer is None:
        tokenizer = model.tokenizer
    
    prompts = [prompt] * num_comparisons
    results = get_n_comparisons(
        prompts=prompts,
        model=model,
        additions=activation_additions,
        tokenizer=tokenizer,
        **kwargs
    )
    
    # Simple text-based printing
    bold = lambda s: f"\033[1m{s}\033[0m"
    
    print("\n" + "=" * 130)
    print(f"| {bold('Unsteered'):^62} | {bold('Steered'):^62} |")
    print("=" * 130)
    
    unsteered = results[results['is_modified'] == False]['completions'].tolist()
    steered = results[results['is_modified'] == True]['completions'].tolist()
    
    for u, s in zip(unsteered, steered):
        u_short = u[:55] + "..." if len(u) > 58 else u
        s_short = s[:55] + "..." if len(s) > 58 else s
        print(f"| {bold(prompt)}{u_short:<{60-len(prompt)}} | {bold(prompt)}{s_short:<{60-len(prompt)}} |")
        print("-" * 130)

