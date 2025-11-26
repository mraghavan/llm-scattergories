import argparse
import torch
from pathlib import Path

from diffusers import AutoPipelineForText2Image, StableDiffusion3Pipeline, CogView4Pipeline

# User-friendly model name mapping
MODEL_DICT = {
    "sd15": "runwayml/stable-diffusion-v1-5",                  # Fast, low memory
    "sdxl": "stabilityai/sdxl-turbo",                          # Best balance
    "flux": "black-forest-labs/FLUX.1-schnell",                # Best quality
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",  # Highest fidelity
    "cogview4": "zai-org/CogView4-6B",                        # Chinese text accuracy, high quality
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
    return parser.parse_args()

args = parse_args()

# Create output directory for generated images
output_dir = Path(__file__).parent / "generated_images"
output_dir.mkdir(exist_ok=True)
print(f"Output directory: {output_dir}")

# Find all .txt files in logos_and_descriptions directory
logos_dir = Path(__file__).parent / "logos_and_descriptions"
txt_files = list(logos_dir.glob("*.txt"))

if not txt_files:
    print(f"No .txt files found in {logos_dir}")
    exit(1)

print(f"Found {len(txt_files)} prompt file(s) in {logos_dir}")

# Generate images for each model
for model_name in args.models:
    MODEL_ID = MODEL_DICT[model_name]
    print(f"\n{'='*60}")
    print(f"Processing model: {model_name} ({MODEL_ID})")
    print(f"{'='*60}")
    print(f"Loading {MODEL_ID}...")
    
    # 1. Load the model
    # We use bfloat16 for most models; SD3 prefers float16 with its native pipeline
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
    else:
        kwargs = {"guidance_scale": 0.0, "num_inference_steps": 2} if "turbo" in MODEL_ID or "schnell" in MODEL_ID else {}
    
    for txt_file in txt_files:
        # Read prompt from .txt file
        with open(txt_file, 'r') as f:
            prompt = f.read().strip()
        
        # Get base name (without .txt extension)
        base_name = txt_file.stem
        
        print(f"\nProcessing prompt file: {txt_file.name}")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        
        # Generate images for each seed
        for seed in range(args.num_images):
            # Use an underscore separator between model name and seed to avoid ambiguity
            # Example: "duolingo_standard_sd3_0.png" instead of "duolingo_standard_sd30.png"
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

