import argparse
import torch
from diffusers import (
    AutoPipelineForText2Image, 
    StableDiffusion3Pipeline, 
    CogView4Pipeline,
    PixArtSigmaPipeline,
    UNet2DConditionModel,
    EulerDiscreteScheduler,
    FluxPipeline
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Compatibility patch for PyTorch < 2.5/2.8: make scaled_dot_product_attention accept enable_gqa
def patch_attention_for_old_pytorch():
    """Patch torch.nn.functional.scaled_dot_product_attention to accept and ignore enable_gqa."""
    try:
        import functools
        
        # Always patch it - it's safe to remove enable_gqa even if PyTorch supports it
        original_sdp = torch.nn.functional.scaled_dot_product_attention
        
        @functools.wraps(original_sdp)
        def patched_sdp(*args, **kwargs):
            # Remove enable_gqa from kwargs if present (not supported in older PyTorch)
            kwargs.pop('enable_gqa', None)
            # Call original function without enable_gqa
            return original_sdp(*args, **kwargs)
        
        torch.nn.functional.scaled_dot_product_attention = patched_sdp
        print("Applied compatibility patch: scaled_dot_product_attention now accepts enable_gqa (ignored)")
    except Exception as e:
        print(f"ERROR: Could not apply attention compatibility patch: {e}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise to prevent continuing with broken state

# Apply the patch before loading models
patch_attention_for_old_pytorch()

# User-friendly model name mapping (only the models we want to count)
MODEL_DICT = {
    "cogview4": "zai-org/CogView4-6B",
    "flux1.dev": "black-forest-labs/FLUX.1-dev",
    "pixart-sigma": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    "sd3.5": "stabilityai/stable-diffusion-3.5-medium",
}

def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_number(num):
    """Format number with appropriate suffix (B for billions, M for millions)."""
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    else:
        return f"{num:,}"

def count_pipeline_parameters(pipe, model_name):
    """Count parameters from all components of a diffusion pipeline."""
    components = {}
    total_all = 0
    trainable_all = 0
    
    # Count parameters from different components
    if hasattr(pipe, 'unet') and pipe.unet is not None:
        total, trainable = count_parameters(pipe.unet)
        components['UNet'] = (total, trainable)
        total_all += total
        trainable_all += trainable
    
    if hasattr(pipe, 'transformer') and pipe.transformer is not None:
        total, trainable = count_parameters(pipe.transformer)
        components['Transformer'] = (total, trainable)
        total_all += total
        trainable_all += trainable
    
    if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
        if isinstance(pipe.text_encoder, (list, tuple)):
            for i, te in enumerate(pipe.text_encoder):
                if te is not None:
                    total, trainable = count_parameters(te)
                    components[f'TextEncoder_{i}'] = (total, trainable)
                    total_all += total
                    trainable_all += trainable
        else:
            total, trainable = count_parameters(pipe.text_encoder)
            components['TextEncoder'] = (total, trainable)
            total_all += total
            trainable_all += trainable
    
    if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
        total, trainable = count_parameters(pipe.text_encoder_2)
        components['TextEncoder2'] = (total, trainable)
        total_all += total
        trainable_all += trainable
    
    if hasattr(pipe, 'vae') and pipe.vae is not None:
        total, trainable = count_parameters(pipe.vae)
        components['VAE'] = (total, trainable)
        total_all += total
        trainable_all += trainable
    
    if hasattr(pipe, 'tokenizer') and pipe.tokenizer is not None:
        # Tokenizers don't have parameters, but we note them
        components['Tokenizer'] = (0, 0)
    
    # For CogView4, also check for other components
    if model_name == "cogview4":
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            # Already counted above
            pass
    
    return components, total_all, trainable_all

def parse_args():
    parser = argparse.ArgumentParser(description="Count parameters in text-to-image models")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_DICT.keys()),
        default=list(MODEL_DICT.keys()),
        help="Model(s) to analyze. Choices: " + ", ".join(MODEL_DICT.keys())
    )
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable sequential CPU offload for memory efficiency"
    )
    return parser.parse_args()

args = parse_args()

print("="*80)
print("Model Parameter Counter")
print("="*80)
print()

results = {}

for model_name in args.models:
    MODEL_ID = MODEL_DICT[model_name]
    print(f"\n{'='*80}")
    print(f"Analyzing: {model_name} ({MODEL_ID})")
    print(f"{'='*80}")
    
    try:
        # Load the model (same logic as sample_images.py)
        if model_name == "sd3.5":
            pipe = StableDiffusion3Pipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
        elif model_name == "cogview4":
            pipe = CogView4Pipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16
            )
        elif model_name == "flux1.dev":
            pipe = FluxPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16
            )
        elif model_name == "pixart-sigma":
            pipe = PixArtSigmaPipeline.from_pretrained(
                MODEL_ID, 
                torch_dtype=torch.float16,
                use_safetensors=True
            )
        else:
            pipe = AutoPipelineForText2Image.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                use_safetensors=True
            )
        
        # Move to device (CPU is fine for counting parameters)
        if args.enable_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipe = pipe.to(device)
        
        # Count parameters
        components, total_all, trainable_all = count_pipeline_parameters(pipe, model_name)
        
        # Store results
        results[model_name] = {
            'components': components,
            'total': total_all,
            'trainable': trainable_all
        }
        
        # Print detailed breakdown
        print(f"\nComponent-wise parameter counts:")
        print(f"{'Component':<20} {'Total Parameters':<25} {'Trainable Parameters':<25}")
        print("-" * 70)
        
        for comp_name, (total, trainable) in components.items():
            if total > 0:  # Only show components with parameters
                print(f"{comp_name:<20} {format_number(total):<25} {format_number(trainable):<25}")
        
        print("-" * 70)
        print(f"{'TOTAL':<20} {format_number(total_all):<25} {format_number(trainable_all):<25}")
        print(f"\nTotal parameters: {total_all:,} ({format_number(total_all)})")
        print(f"Trainable parameters: {trainable_all:,} ({format_number(trainable_all)})")
        
        # Clean up
        del pipe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"ERROR loading {model_name}: {e}")
        import traceback
        traceback.print_exc()
        results[model_name] = {'error': str(e)}

# Print summary
print(f"\n\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"{'Model':<20} {'Total Parameters':<25}")
print("-" * 45)
for model_name in args.models:
    if model_name in results and 'error' not in results[model_name]:
        total = results[model_name]['total']
        print(f"{model_name:<20} {format_number(total):<25} ({total:,})")
    else:
        print(f"{model_name:<20} {'ERROR':<25}")

print(f"\n{'='*80}")

