import argparse
import torch
from pathlib import Path

from datasets import load_dataset
from diffusers import AutoPipelineForText2Image, StableDiffusion3Pipeline, CogView4Pipeline

# User-friendly model name mapping
MODEL_DICT = {
    "sd15": "runwayml/stable-diffusion-v1-5",                  # Fast, low memory
    "sdxl": "stabilityai/sdxl-turbo",                          # Best balance
    "flux": "black-forest-labs/FLUX.1-schnell",                # Best quality
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
        default=None,
        help="Maximum number of samples from dataset to process (default: all)"
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

# Limit number of samples if specified
if args.max_samples is not None:
    dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    print(f"Processing {len(dataset)} samples (limited by --max-samples)")

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
    
    # 2. Move model to GPU
    # Move the entire pipeline to CUDA for GPU acceleration
    if model_name == "cogview4":
        # CogView4 benefits from CPU offload for memory efficiency
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    else:
        pipe = pipe.to("cuda")
    
    # Optional: Helps with memory for SDXL/Flux at slight speed cost
    # pipe.enable_attention_slicing() 
    
    # 3. Generate images for each prompt file
    # Note: "guidance_scale=0.0" is specific to Turbo/Schnell models. 
    # For standard SD1.5/SDXL, remove it or set it to 7.5
    if model_name == "sd3":
        kwargs = {"guidance_scale": 5.0, "num_inference_steps": 28}
    elif model_name == "cogview4":
        kwargs = {"guidance_scale": 3.5, "num_inference_steps": 50}
    elif model_name == "playground":
        kwargs = {"guidance_scale": 3.0, "num_inference_steps": 50}
    else:
        kwargs = {"guidance_scale": 0.0, "num_inference_steps": 2} if "turbo" in MODEL_ID or "schnell" in MODEL_ID else {}
    
    for idx, sample in enumerate(dataset):
        # Get prompt from caption field
        prompt = sample["caption"].strip()
        
        # Create base name from dataset index
        base_name = f"ios_icon_{idx}"
        
        print(f"\nProcessing sample {idx + 1}/{len(dataset)}: {base_name}")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        
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

