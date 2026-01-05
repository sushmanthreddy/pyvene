"""
Activation Addition (ActAdd) Utilities for pyvene

This module implements the Activation Addition technique for steering language model
outputs at inference time. ActAdd works by:
1. Computing a "steering vector" from contrasting prompts (e.g., "Love" vs "Hate")
2. Adding this vector to the model's residual stream during inference

Reference: "Activation Addition: Steering Language Models Without Optimization"

Example usage:
    >>> from pyvene.models.actadd import compute_steering_vector, create_actadd_model
    >>> 
    >>> # Compute steering vector from contrasting prompts
    >>> steering_vector = compute_steering_vector(
    ...     model, tokenizer,
    ...     positive_prompt="Love",
    ...     negative_prompt="Hate",
    ...     layer=6
    ... )
    >>> 
    >>> # Create intervenable model with ActAdd
    >>> actadd_model = create_actadd_model(
    ...     model,
    ...     steering_vector=steering_vector,
    ...     layer=6,
    ...     coeff=5.0
    ... )
    >>> 
    >>> # Generate with steering
    >>> output = actadd_model.generate(
    ...     tokenizer("I hate you because", return_tensors="pt"),
    ...     max_new_tokens=50
    ... )
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union, Dict, Tuple
from dataclasses import dataclass

from .interventions import AdditionIntervention, ConstantSourceIntervention
from .intervenable_base import IntervenableModel
from .configuration_intervenable_model import IntervenableConfig, RepresentationConfig


@dataclass
class SteeringVector:
    """
    A steering vector computed from contrasting prompts.
    
    Attributes:
        vector: The steering vector tensor of shape (1, seq_len, hidden_dim)
        positive_prompt: The positive/target prompt used
        negative_prompt: The negative/baseline prompt used
        layer: The layer where this vector was extracted
        coeff: Default coefficient for steering strength
    """
    vector: torch.Tensor
    positive_prompt: str
    negative_prompt: str
    layer: int
    coeff: float = 1.0
    
    def __repr__(self):
        return (
            f"SteeringVector(layer={self.layer}, coeff={self.coeff}, "
            f"shape={tuple(self.vector.shape)}, "
            f"prompts='{self.positive_prompt}' - '{self.negative_prompt}')"
        )
    
    def scaled(self, coeff: float) -> torch.Tensor:
        """Return the steering vector scaled by coefficient."""
        return self.coeff * coeff * self.vector
    
    def to(self, device: Union[str, torch.device]) -> "SteeringVector":
        """Move steering vector to device."""
        return SteeringVector(
            vector=self.vector.to(device),
            positive_prompt=self.positive_prompt,
            negative_prompt=self.negative_prompt,
            layer=self.layer,
            coeff=self.coeff
        )


class ActAddIntervention(AdditionIntervention, ConstantSourceIntervention):
    """
    Activation Addition intervention with a precomputed steering vector.
    
    This intervention adds a constant steering vector to the base activations,
    implementing the ActAdd technique for model steering.
    
    Args:
        steering_vector: The precomputed steering vector to add
        coeff: Scaling coefficient for the steering vector (default: 1.0)
        **kwargs: Additional arguments passed to parent intervention classes
    """
    
    def __init__(self, steering_vector: torch.Tensor, coeff: float = 1.0, **kwargs):
        # Scale the vector by coefficient
        scaled_vector = coeff * steering_vector
        kwargs["source_representation"] = scaled_vector
        super().__init__(**kwargs)
        self.coeff = coeff
        self.original_steering_vector = steering_vector
        
    def set_coeff(self, coeff: float):
        """Update the steering coefficient."""
        self.coeff = coeff
        self.source_representation = coeff * self.original_steering_vector
        
    def __str__(self):
        return f"ActAddIntervention(coeff={self.coeff})"


def get_model_layers(model: nn.Module) -> nn.ModuleList:
    """
    Automatically detect the transformer layers in a HuggingFace model.
    
    Works by finding the ModuleList containing >50% of model parameters,
    which is typically the transformer block stack.
    
    Args:
        model: A HuggingFace transformer model
        
    Returns:
        The ModuleList containing transformer layers
    """
    def numel(mod):
        return sum(p.numel() for p in mod.parameters())
    
    model_numel = numel(model)
    candidates = [
        mod for mod in model.modules()
        if isinstance(mod, nn.ModuleList) and numel(mod) > 0.5 * model_numel
    ]
    
    if len(candidates) != 1:
        raise ValueError(
            f"Could not automatically detect transformer layers. "
            f"Found {len(candidates)} candidates with >50% of model params."
        )
    
    return candidates[0]


def get_num_layers(model: nn.Module) -> int:
    """Get the number of transformer layers in the model."""
    return len(get_model_layers(model))


@torch.no_grad()
def extract_activations(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    layer: int,
    component: str = "block_output",
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """
    Extract activations from a model at a specific layer.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompts to extract activations for
        layer: Which layer to extract from (0-indexed)
        component: Which component to extract ("block_output", "mlp_output", etc.)
        device: Device to run on (defaults to model's device)
        
    Returns:
        Tensor of activations with shape (batch, seq_len, hidden_dim)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Tokenize inputs
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Use model's native hidden states output (most reliable)
    outputs = model(**inputs, output_hidden_states=True)
    
    # hidden_states is a tuple of (num_layers + 1,) tensors
    # Index 0 is embeddings, index 1 is after layer 0, etc.
    # So layer N's output is at index N+1
    hidden_states = outputs.hidden_states
    
    # Get activations at the specified layer (after that layer's processing)
    # layer=0 means output of first transformer block, which is hidden_states[1]
    activations = hidden_states[layer + 1]
    
    return activations


def compute_steering_vector(
    model: nn.Module,
    tokenizer,
    positive_prompt: str,
    negative_prompt: str,
    layer: int,
    component: str = "block_output",
    coeff: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
    position: Optional[int] = None,
) -> SteeringVector:
    """
    Compute a steering vector from contrasting prompts.
    
    The steering vector is computed as:
        steering_vector = activation(positive_prompt) - activation(negative_prompt)
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        positive_prompt: The target/positive prompt (e.g., "Love")
        negative_prompt: The baseline/negative prompt (e.g., "Hate")
        layer: Which layer to extract the vector from (0-indexed)
        component: Which component ("block_output", "mlp_output", "attention_output")
        coeff: Default scaling coefficient
        device: Device to run on
        position: If specified, only use activation at this position
        
    Returns:
        A SteeringVector object containing the computed vector
        
    Example:
        >>> sv = compute_steering_vector(
        ...     model, tokenizer,
        ...     positive_prompt="Love",
        ...     negative_prompt="Hate",
        ...     layer=6,
        ...     coeff=5.0
        ... )
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Extract activations for both prompts
    activations = extract_activations(
        model, tokenizer,
        prompts=[positive_prompt, negative_prompt],
        layer=layer,
        component=component,
        device=device,
    )
    
    # Compute difference: positive - negative
    positive_act = activations[0]  # (seq_len, hidden_dim)
    negative_act = activations[1]  # (seq_len, hidden_dim)
    
    # Handle different sequence lengths by taking minimum
    min_len = min(positive_act.shape[0], negative_act.shape[0])
    steering_vec = positive_act[:min_len] - negative_act[:min_len]
    
    # Optionally select specific position
    if position is not None:
        steering_vec = steering_vec[position:position+1]
    
    # Add batch dimension
    steering_vec = steering_vec.unsqueeze(0)
    
    return SteeringVector(
        vector=steering_vec,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        layer=layer,
        coeff=coeff
    )


def create_actadd_model(
    model: nn.Module,
    steering_vector: Union[SteeringVector, torch.Tensor],
    layer: Optional[int] = None,
    coeff: float = 1.0,
    component: str = "block_output",
    position: Optional[int] = None,
    prompt_length: Optional[int] = None,
) -> IntervenableModel:
    """
    Create an IntervenableModel configured for Activation Addition.
    
    Args:
        model: The base language model
        steering_vector: Either a SteeringVector object or raw tensor
        layer: Layer to intervene at (required if steering_vector is a tensor)
        coeff: Scaling coefficient for the steering vector
        component: Component to intervene on ("block_output", "mlp_output", etc.)
        position: If specified, only intervene at this token position
        prompt_length: Length of the prompt (for padding the steering vector)
        
    Returns:
        An IntervenableModel configured for ActAdd steering
        
    Example:
        >>> actadd_model = create_actadd_model(
        ...     model,
        ...     steering_vector=sv,  # From compute_steering_vector
        ...     coeff=5.0,
        ...     prompt_length=5
        ... )
        >>> 
        >>> # Generate with steering
        >>> _, output = actadd_model.generate(
        ...     tokenizer("I hate you because", return_tensors="pt"),
        ...     max_new_tokens=50,
        ...     intervene_on_prompt=True
        ... )
    """
    # Extract info from SteeringVector if provided
    if isinstance(steering_vector, SteeringVector):
        vec = steering_vector.vector.clone()
        layer = steering_vector.layer
        coeff = coeff * steering_vector.coeff  # Combine coefficients
    else:
        vec = steering_vector.clone() if hasattr(steering_vector, 'clone') else steering_vector
        if layer is None:
            raise ValueError("layer must be specified when steering_vector is a tensor")
    
    # Ensure proper shape (1, seq_len, hidden_dim)
    if len(vec.shape) == 2:
        vec = vec.unsqueeze(0)
    
    # Apply coefficient to create the source representation
    scaled_vec = coeff * vec
    
    # Pad steering vector to match prompt length if needed
    if prompt_length is not None and scaled_vec.shape[1] < prompt_length:
        padding_size = prompt_length - scaled_vec.shape[1]
        padding = torch.zeros(
            scaled_vec.shape[0], padding_size, scaled_vec.shape[2],
            device=scaled_vec.device, dtype=scaled_vec.dtype
        )
        scaled_vec = torch.cat([scaled_vec, padding], dim=1)
    
    # Get sequence length from steering vector
    seq_len = scaled_vec.shape[1]
    
    # Create config with AdditionIntervention and constant source
    config = IntervenableConfig(
        representations=[
            RepresentationConfig(
                layer=layer,
                component=component,
                unit="pos",
                max_number_of_units=seq_len,
                source_representation=scaled_vec,
            )
        ],
        intervention_types=AdditionIntervention,
    )
    
    return IntervenableModel(config, model)


def create_multi_layer_actadd_model(
    model: nn.Module,
    steering_vectors: List[Union[SteeringVector, Tuple[torch.Tensor, int]]],
    coeff: float = 1.0,
    component: str = "block_output",
    prompt_length: Optional[int] = None,
) -> IntervenableModel:
    """
    Create an IntervenableModel with ActAdd at multiple layers.
    
    This allows stacking multiple steering vectors at different layers,
    which can create more complex steering effects.
    
    Args:
        model: The base language model
        steering_vectors: List of SteeringVector objects or (tensor, layer) tuples
        coeff: Global scaling coefficient applied to all vectors
        component: Component to intervene on
        
    Returns:
        An IntervenableModel with multiple ActAdd interventions
        
    Example:
        >>> sv1 = compute_steering_vector(model, tok, "Love", "Hate", layer=6)
        >>> sv2 = compute_steering_vector(model, tok, "Happy", "Sad", layer=12)
        >>> 
        >>> multi_actadd = create_multi_layer_actadd_model(
        ...     model,
        ...     steering_vectors=[sv1, sv2],
        ...     coeff=3.0
        ... )
    """
    representations = []
    intervention_types = []
    
    for sv in steering_vectors:
        if isinstance(sv, SteeringVector):
            vec = sv.vector.clone()
            layer = sv.layer
            sv_coeff = sv.coeff
        else:
            vec, layer = sv
            vec = vec.clone() if hasattr(vec, 'clone') else vec
            sv_coeff = 1.0
        
        # Ensure proper shape
        if len(vec.shape) == 2:
            vec = vec.unsqueeze(0)
        
        # Apply combined coefficient
        total_coeff = coeff * sv_coeff
        scaled_vec = total_coeff * vec
        
        # Pad steering vector to match prompt length if needed
        if prompt_length is not None and scaled_vec.shape[1] < prompt_length:
            padding_size = prompt_length - scaled_vec.shape[1]
            padding = torch.zeros(
                scaled_vec.shape[0], padding_size, scaled_vec.shape[2],
                device=scaled_vec.device, dtype=scaled_vec.dtype
            )
            scaled_vec = torch.cat([scaled_vec, padding], dim=1)
        
        representations.append(
            RepresentationConfig(
                layer=layer,
                component=component,
                unit="pos",
                max_number_of_units=scaled_vec.shape[1],
                source_representation=scaled_vec,
            )
        )
        intervention_types.append(AdditionIntervention)
    
    config = IntervenableConfig(
        representations=representations,
        intervention_types=intervention_types,
    )
    return IntervenableModel(config, model)


def generate_with_steering(
    model: nn.Module,
    tokenizer,
    prompt: str,
    steering_vector: Union[SteeringVector, torch.Tensor],
    layer: Optional[int] = None,
    coeff: float = 1.0,
    max_new_tokens: int = 50,
    component: str = "block_output",
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.9,
    **generate_kwargs,
) -> Tuple[str, str]:
    """
    Generate text with and without steering for comparison.
    
    This is a convenience function that creates the ActAdd model,
    generates both steered and unsteered outputs, and returns decoded text.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The input prompt
        steering_vector: The steering vector (SteeringVector or tensor)
        layer: Layer to intervene at (required if vector is tensor)
        coeff: Steering coefficient
        max_new_tokens: Maximum tokens to generate
        component: Component to intervene on
        do_sample: Whether to use sampling
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        **generate_kwargs: Additional arguments for model.generate()
        
    Returns:
        Tuple of (unsteered_output, steered_output) as decoded strings
        
    Example:
        >>> unsteered, steered = generate_with_steering(
        ...     model, tokenizer,
        ...     prompt="I hate you because",
        ...     steering_vector=love_hate_vector,
        ...     coeff=5.0,
        ...     max_new_tokens=50
        ... )
        >>> print("Unsteered:", unsteered)
        >>> print("Steered:", steered)
    """
    device = next(model.parameters()).device
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get prompt length for padding
    prompt_length = inputs["input_ids"].shape[1]
    
    # Get layer from SteeringVector if needed
    if isinstance(steering_vector, SteeringVector):
        layer = steering_vector.layer
        steering_vector = steering_vector.to(device)
    
    # Generation kwargs
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        **generate_kwargs
    }
    
    # Generate unsteered output
    with torch.no_grad():
        unsteered_ids = model.generate(**inputs, **gen_kwargs)
    unsteered_text = tokenizer.decode(unsteered_ids[0], skip_special_tokens=True)
    
    # Create ActAdd model and generate steered output
    actadd_model = create_actadd_model(
        model,
        steering_vector=steering_vector,
        layer=layer,
        coeff=coeff,
        component=component,
        prompt_length=prompt_length,  # Pass prompt length for padding
    )
    
    with torch.no_grad():
        _, steered_ids = actadd_model.generate(
            inputs,
            intervene_on_prompt=True,  # Only intervene on prompt, not generated tokens
            **gen_kwargs
        )
    steered_text = tokenizer.decode(steered_ids[0], skip_special_tokens=True)
    
    return unsteered_text, steered_text


def compare_generations(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    steering_vector: Union[SteeringVector, torch.Tensor],
    layer: Optional[int] = None,
    coeff: float = 1.0,
    num_samples: int = 3,
    max_new_tokens: int = 50,
    **generate_kwargs,
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Generate multiple samples for multiple prompts, comparing steered vs unsteered.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of prompts to test
        steering_vector: The steering vector
        layer: Layer to intervene at
        coeff: Steering coefficient
        num_samples: Number of samples per prompt
        max_new_tokens: Maximum tokens to generate
        **generate_kwargs: Additional generation arguments
        
    Returns:
        Dict mapping each prompt to a list of (unsteered, steered) tuples
    """
    results = {}
    
    for prompt in prompts:
        samples = []
        for _ in range(num_samples):
            unsteered, steered = generate_with_steering(
                model, tokenizer, prompt,
                steering_vector=steering_vector,
                layer=layer,
                coeff=coeff,
                max_new_tokens=max_new_tokens,
                **generate_kwargs
            )
            samples.append((unsteered, steered))
        results[prompt] = samples
    
    return results


def print_comparison(
    results: Dict[str, List[Tuple[str, str]]],
    prompt_only: bool = False,
):
    """
    Pretty-print the comparison results.
    
    Args:
        results: Output from compare_generations()
        prompt_only: If True, only show the generated part (not the prompt)
    """
    for prompt, samples in results.items():
        print("=" * 80)
        print(f"PROMPT: {prompt}")
        print("=" * 80)
        
        for i, (unsteered, steered) in enumerate(samples, 1):
            if prompt_only:
                unsteered = unsteered[len(prompt):]
                steered = steered[len(prompt):]
            
            print(f"\n--- Sample {i} ---")
            print(f"Unsteered: {unsteered}")
            print(f"Steered:   {steered}")
        print()

