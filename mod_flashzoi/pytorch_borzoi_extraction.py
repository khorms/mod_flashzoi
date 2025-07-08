import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any

def extract_all_representations_single_pass(model, x: torch.Tensor) -> Dict[str, Any]:
    """
    Extract ALL types of representations in a single forward pass
    
    Args:
        model: Borzoi model
        x: Input sequence tensor [4, seq_len] or [batch, 4, seq_len]
        
    Returns:
        Dictionary containing all intermediate representations
    """
    model.eval()
    
    # Storage for all intermediate results
    layer_hidden_states = {}      # Method 2: positional representations
    attention_weights = {}        # Method 1: attention matrices  
    value_vectors = {}           # Method 1: value vectors
    head_outputs = {}            # Method 3: per-head outputs
    
    def transformer_layer_hook(layer_idx):
        def hook(module, input, output):
            # Capture full layer output (Method 2)
            layer_hidden_states[f'layer_{layer_idx}'] = output.clone()
        return hook
    
    def attention_module_hook(layer_idx):
        def hook(module, input, output):
            # Capture attention intermediate values (Methods 1 & 3)
            if hasattr(module, 'last_attention_weights') and module.last_attention_weights is not None:
                attention_weights[f'layer_{layer_idx}'] = module.last_attention_weights.clone()
            
            if hasattr(module, 'last_value_vectors') and module.last_value_vectors is not None:
                value_vectors[f'layer_{layer_idx}'] = module.last_value_vectors.clone()
                
            if hasattr(module, 'last_head_outputs') and module.last_head_outputs is not None:
                head_outputs[f'layer_{layer_idx}'] = module.last_head_outputs.clone()
        return hook
    
    # Register hooks and enable storage
    layer_hooks = []
    attention_hooks = []
    
    for i, layer in enumerate(model.transformer):
        # Hook the full transformer layer (for Method 2)
        layer_hook = layer.register_forward_hook(transformer_layer_hook(i))
        layer_hooks.append(layer_hook)
        
        # Hook the attention module specifically (for Methods 1 & 3)
        attention_module = layer[0].fn[1]  # Navigate to Attention module
        
        # Enable storage for this forward pass
        if hasattr(attention_module, 'enable_intermediate_storage'):
            attention_module.enable_intermediate_storage()
            
        attention_hook = attention_module.register_forward_hook(attention_module_hook(i))
        attention_hooks.append(attention_hook)
    
    # Single forward pass captures everything!
    with torch.no_grad():
        x_input = x.unsqueeze(0) if x.dim() == 2 else x  # Ensure batch dim
        
        # Standard forward pass through the model
        x_conv = model.conv_dna(x_input)
        x_unet0 = model.res_tower(x_conv)
        x_unet1 = model.unet1(x_unet0)
        x_transformer_input = model._max_pool(x_unet1)
        x_transformer_input = x_transformer_input.permute(0, 2, 1)
        
        # This single call captures ALL intermediate representations
        final_output = model.transformer(x_transformer_input)
    
    # Clean up hooks and disable storage
    for hook in layer_hooks + attention_hooks:
        hook.remove()
    
    for layer in model.transformer:
        attention_module = layer[0].fn[1]
        if hasattr(attention_module, 'disable_intermediate_storage'):
            attention_module.disable_intermediate_storage()
    
    # Package results
    batch_size, seq_len, hidden_dim = final_output.shape
    
    results = {
        'layer_hidden_states': layer_hidden_states,  # Method 2: [batch, seq_len, hidden_dim]
        'attention_weights': attention_weights,      # Method 1: [batch, heads, seq_len, seq_len] 
        'value_vectors': value_vectors,              # Method 1: [batch, heads, seq_len, dim_value]
        'head_outputs': head_outputs,                # Method 3: [batch, heads, seq_len, dim_value]
        'final_output': final_output,
        'sequence_info': {
            'batch_size': batch_size,
            'seq_len': seq_len, 
            'hidden_dim': hidden_dim
        }
    }
    
    return results

def extract_method1_attention_weighted(all_results: Dict, target_bin_idx: int) -> Dict[str, torch.Tensor]:
    """
    Method 1: Attention-weighted pooling for specific bin
    
    Args:
        all_results: Output from extract_all_representations_single_pass
        target_bin_idx: Which bin to extract features for
        
    Returns:
        Dictionary of attention-weighted representations per layer
    """
    bin_features = {}
    
    for layer_name in all_results['attention_weights'].keys():
        attn_weights = all_results['attention_weights'][layer_name]  # [batch, heads, seq_len, seq_len]
        value_vectors = all_results['value_vectors'][layer_name]     # [batch, heads, seq_len, dim_value]
        
        # Extract attention from target_bin to all positions
        bin_attention = attn_weights[0, :, target_bin_idx, :]  # [heads, seq_len]
        
        # Average attention across heads
        avg_attention = bin_attention.mean(dim=0)  # [seq_len]
        
        # Get value vectors (average across heads)
        avg_values = value_vectors[0].mean(dim=0)  # [seq_len, dim_value]
        
        # Compute attention-weighted representation
        weighted_repr = torch.sum(avg_attention.unsqueeze(1) * avg_values, dim=0)  # [dim_value]
        
        bin_features[layer_name] = weighted_repr
    
    return bin_features

def extract_method2_positional(all_results: Dict, target_bin_idx: int) -> Dict[str, torch.Tensor]:
    """
    Method 2: Direct positional extraction (most straightforward)
    
    Args:
        all_results: Output from extract_all_representations_single_pass
        target_bin_idx: Which bin to extract features for
        
    Returns:
        Dictionary of positional representations per layer
    """
    bin_features = {}
    
    for layer_name, hidden_states in all_results['layer_hidden_states'].items():
        # Extract representation for target bin
        bin_features[layer_name] = hidden_states[0, target_bin_idx, :]  # [hidden_dim]
    
    return bin_features

def extract_method3_multihead(all_results: Dict, target_bin_idx: int) -> Dict[str, Dict]:
    """
    Method 3: Multi-head decomposition
    
    Args:
        all_results: Output from extract_all_representations_single_pass
        target_bin_idx: Which bin to extract features for
        
    Returns:
        Dictionary with per-head and combined representations per layer
    """
    bin_features = {}
    
    for layer_name, head_outputs in all_results['head_outputs'].items():
        # head_outputs shape: [batch, heads, seq_len, dim_value]
        bin_head_outputs = head_outputs[0, :, target_bin_idx, :]  # [heads, dim_value]
        
        bin_features[layer_name] = {
            'per_head': bin_head_outputs,                           # [heads, dim_value]
            'combined': bin_head_outputs.flatten(),                 # [heads * dim_value]
            'mean_across_heads': bin_head_outputs.mean(dim=0),      # [dim_value]
            'num_heads': bin_head_outputs.shape[0],
            'head_dim': bin_head_outputs.shape[1]
        }
    
    return bin_features

def prepare_classifier_features_all_methods(all_results: Dict, method: str = 'positional') -> torch.Tensor:
    """
    Prepare features for ALL bins for classification
    
    Args:
        all_results: Output from extract_all_representations_single_pass
        method: 'positional', 'attention_weighted', or 'multihead'
        
    Returns:
        Feature tensor [num_bins, feature_dim]
    """
    if method == 'positional':
        # Method 2: Concatenate hidden states from all layers
        layer_names = sorted([k for k in all_results['layer_hidden_states'].keys() if k.startswith('layer_')])
        layer_features = []
        
        for layer_name in layer_names:
            layer_features.append(all_results['layer_hidden_states'][layer_name][0])  # [seq_len, hidden_dim]
        
        # Concatenate along feature dimension: [seq_len, num_layers * hidden_dim]
        combined_features = torch.cat(layer_features, dim=1)
        
    elif method == 'multihead':
        # Method 3: Use per-head representations
        layer_names = sorted([k for k in all_results['head_outputs'].keys() if k.startswith('layer_')])
        layer_features = []
        
        for layer_name in layer_names:
            head_outputs = all_results['head_outputs'][layer_name][0]  # [heads, seq_len, dim_value]
            # Reshape to [seq_len, heads * dim_value]
            reshaped = head_outputs.permute(1, 0, 2).flatten(start_dim=1)
            layer_features.append(reshaped)
        
        # Concatenate along feature dimension
        combined_features = torch.cat(layer_features, dim=1)
        
    else:
        raise ValueError(f"Method '{method}' not implemented for all-bins extraction")
    
    return combined_features