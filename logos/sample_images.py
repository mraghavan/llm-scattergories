import argparse
import torch
from pathlib import Path

from datasets import load_dataset
from diffusers import AutoPipelineForText2Image, StableDiffusion3Pipeline, CogView4Pipeline
from PIL import Image

# User-friendly model name mapping
MODEL_DICT = {
    "sd15": "runwayml/stable-diffusion-v1-5",                  # Fast, low memory
    "sdxl": "stabilityai/sdxl-turbo",                          # Best balance
    "flux": "black-forest-labs/FLUX.1-schnell",                # Best quality
    "flux1.dev": "black-forest-labs/FLUX.1-dev",               # FLUX.1-dev model
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",  # Highest fidelity
    "cogview4": "zai-org/CogView4-6B",                        # Chinese text accuracy, high quality
    "playground": "playgroundai/playground-v2.5-1024px-aesthetic",  # High aesthetic quality (only v2.5 variant available)
}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using text-to-image models")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_DICT.keys()),
        default=["sd15"],
        help="Model(s) to use. Choices: " + ", ".join(MODEL_DICT.keys())
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate (SEED will iterate from 0 to num-images-1)"
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to use (default: 'train')"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Maximum number of samples from dataset to process (default: 5)"
    )
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable sequential CPU offload for memory efficiency (slower but uses less VRAM)"
    )
    parser.add_argument(
        "--enable-compile",
        action="store_true",
        help="Enable torch.compile() for faster inference (requires PyTorch 2.0+)"
    )
    parser.add_argument(
        "--enable-attention-slicing",
        action="store_true",
        help="Enable attention slicing to reduce memory usage (slight speed cost)"
    )
    return parser.parse_args()

args = parse_args()

# Create output directory for generated images
output_dir = Path(__file__).parent / "generated_images"
output_dir.mkdir(exist_ok=True)
print(f"Output directory: {output_dir}")

# Load the iOS app icons dataset from Hugging Face
print("Loading iOS app icons dataset from Hugging Face...")
dataset = load_dataset("ppierzc/ios-app-icons", split=args.dataset_split)
print(f"Loaded {len(dataset)} samples from dataset (split: {args.dataset_split})")

# Apply deterministic shuffle with seed 0
print("Applying deterministic shuffle (seed=0)...")
dataset = dataset.shuffle(seed=0)

# Limit number of samples
dataset = dataset.select(range(min(args.max_samples, len(dataset))))
print(f"Processing {len(dataset)} samples (limited by --max-samples={args.max_samples})")

# Verify dataset has required fields
if "caption" not in dataset.column_names:
    raise ValueError("Dataset must have a 'caption' field")
if "image" not in dataset.column_names:
    raise ValueError("Dataset must have an 'image' field")

# Generate images for each model
for model_name in args.models:
    MODEL_ID = MODEL_DICT[model_name]
    print(f"\n{'='*60}")
    print(f"Processing model: {model_name} ({MODEL_ID})")
    print(f"{'='*60}")
    print(f"Loading {MODEL_ID}...")
    
    # 1. Load the model
    # We use bfloat16 for most models; SD3 and Playground prefer float16
    if model_name == "sd3":
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
    elif model_name == "playground":
        pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
    else:
        pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        )
    
    # 2. Move model to GPU and apply optimizations
    # Enable VAE optimizations for all models (helps with 1024x1024 resolution)
    if hasattr(pipe, 'vae'):
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    
    # Memory management: CPU offload or direct GPU
    if model_name == "cogview4" or args.enable_cpu_offload:
        # Sequential CPU offload: moves model components between CPU/GPU as needed
        # Slower but uses less VRAM - useful for large models or limited GPU memory
        pipe.enable_model_cpu_offload()
        print("  Enabled sequential CPU offload")
    else:
        pipe = pipe.to("cuda")
    
    # Attention slicing: reduces memory at slight speed cost
    if args.enable_attention_slicing:
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing()
            print("  Enabled attention slicing")
    
    # Model compilation: can significantly speed up inference (PyTorch 2.0+)
    if args.enable_compile:
        try:
            if hasattr(torch, 'compile'):
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
                if hasattr(pipe, 'transformer') and pipe.transformer is not None:
                    pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
                print("  Enabled torch.compile() optimization")
            else:
                print("  Warning: torch.compile() not available (requires PyTorch 2.0+)")
        except Exception as e:
            print(f"  Warning: torch.compile() failed: {e}")
    
    # Warmup run for compiled models (first inference is slower)
    if args.enable_compile and hasattr(torch, 'compile'):
        print("  Running warmup inference...")
        try:
            warmup_generator = torch.Generator(device="cuda").manual_seed(0)
            warmup_kwargs = kwargs.copy()
            if model_name == "sd3":
                warmup_kwargs = {"guidance_scale": 5.0, "num_inference_steps": 1}  # Faster warmup
            elif model_name in ["cogview4", "playground"]:
                warmup_kwargs = {"guidance_scale": 3.0, "num_inference_steps": 1}  # Faster warmup
            else:
                warmup_kwargs = {"guidance_scale": 0.0, "num_inference_steps": 1} if "turbo" in MODEL_ID or "schnell" in MODEL_ID else {"num_inference_steps": 1}
            
            _ = pipe(
                prompt="warmup",
                generator=warmup_generator,
                width=1024,
                height=1024,
                **warmup_kwargs
            ).images[0]
            print("  Warmup complete")
        except Exception as e:
            print(f"  Warning: Warmup failed: {e}")
    
    # 3. Generate images for each prompt file
    # Note: "guidance_scale=0.0" is specific to Turbo/Schnell models. 
    # For standard SD1.5/SDXL, remove it or set it to 7.5
    if model_name == "sd3":
        kwargs = {"guidance_scale": 5.0, "num_inference_steps": 28}
    elif model_name == "cogview4":
        kwargs = {"guidance_scale": 3.5, "num_inference_steps": 50}
    elif model_name == "playground":
        kwargs = {"guidance_scale": 3.0, "num_inference_steps": 50}
    elif model_name == "flux1.dev":
        # FLUX.1-dev: higher quality, more steps than schnell variant
        kwargs = {"guidance_scale": 3.5, "num_inference_steps": 28}
    else:
        kwargs = {"guidance_scale": 0.0, "num_inference_steps": 2} if "turbo" in MODEL_ID or "schnell" in MODEL_ID else {}
    
    for idx, sample in enumerate(dataset):
        # Get prompt from caption field
        prompt = sample["caption"].strip()
        
        # Create base name from dataset index
        base_name = f"ios_icon_{idx}"
        
        print(f"\nProcessing sample {idx + 1}/{len(dataset)}: {base_name}")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        
        # Save original image for evaluation
        original_filename = output_dir / f"{base_name}_original.png"
        if not original_filename.exists():
            original_image = sample["image"]
            if not isinstance(original_image, Image.Image):
                original_image = Image.fromarray(original_image)
            original_image = original_image.convert("RGB")
            original_image.save(original_filename)
            print(f"  Saved original image to {original_filename.name}")
        
        # Generate images for each seed
        for seed in range(args.num_images):
            # Use an underscore separator between model name and seed to avoid ambiguity
            # Example: "ios_icon_0_sd3_0.png" instead of "ios_icon_0_sd30.png"
            output_filename = output_dir / f"{base_name}_{model_name}_{seed}.png"
            
            # Skip if file already exists
            if output_filename.exists():
                print(f"  Skipping {output_filename.name} (already exists)")
                continue
            
            print(f"  Generating image {seed + 1}/{args.num_images} with seed={seed}...")
            generator = torch.Generator(device="cuda").manual_seed(seed)
            
            # Use 1024x1024 for all models for fair comparison
            # (SD1.5 defaults to 512x512, but we standardize to 1024x1024)
            # CogView4 requires explicit width/height (must be divisible by 32, between 512-2048)
            image = pipe(
                prompt=prompt,
                generator=generator,
                width=1024,
                height=1024,
                **kwargs
            ).images[0]
            
            image.save(output_filename)
            print(f"  Saved to {output_filename}")
    
    # Clean up model to free GPU memory before loading next model
    del pipe
    torch.cuda.empty_cache()
    print(f"Completed model: {model_name}")

print(f"\n{'='*60}")
print("All generations complete!")
print(f"{'='*60}")

