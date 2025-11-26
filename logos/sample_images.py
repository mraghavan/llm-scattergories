import argparse
import torch
from pathlib import Path
from PIL import Image

from diffusers import AutoPipelineForText2Image, StableDiffusion3Pipeline

# User-friendly model name mapping
MODEL_DICT = {
    "sd15": "runwayml/stable-diffusion-v1-5",                  # Fast, low memory
    "sdxl": "stabilityai/sdxl-turbo",                          # Best balance
    "flux": "black-forest-labs/FLUX.1-schnell",                # Best quality
    "sd3": "stabilityai/stable-diffusion-3-medium-diffusers",  # Highest fidelity
    "janus": "deepseek-ai/Janus-Pro-7B",                       # DeepSeek Janus-Pro-7B (uses transformers, not diffusers)
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
    # Janus-Pro-7B uses a different architecture and requires transformers, not diffusers
    is_janus = (model_name == "janus")
    
    if is_janus:
        # Try to import transformers for Janus
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError:
            print(f"\nERROR: Janus-Pro-7B requires the 'transformers' library.")
            print("Please install it with: pip install transformers")
            continue
    MODEL_ID = MODEL_DICT[model_name]
    print(f"\n{'='*60}")
    print(f"Processing model: {model_name} ({MODEL_ID})")
    print(f"{'='*60}")
    print(f"Loading {MODEL_ID}...")
    
    # 1. Load the model
    if is_janus:
        # Janus-Pro-7B uses transformers library with a different architecture
        try:
            processor = AutoProcessor.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            print("Janus-Pro-7B loaded successfully")
        except Exception as e:
            print(f"ERROR: Failed to load Janus-Pro-7B: {e}")
            print("Note: Janus-Pro-7B may require specific setup. Check DeepSeek's documentation.")
            continue
    else:
        # Standard diffusers models
        # We use bfloat16 for most models; SD3 prefers float16 with its native pipeline
        if model_name == "sd3":
            pipe = StableDiffusion3Pipeline.from_pretrained(
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
        
        # 2. Move model to GPU
        # Move the entire pipeline to CUDA for GPU acceleration
        pipe = pipe.to("cuda")
        
        # Optional: Helps with memory for SDXL/Flux at slight speed cost
        # pipe.enable_attention_slicing() 
        
        # 3. Set generation parameters for diffusers models
        # Note: "guidance_scale=0.0" is specific to Turbo/Schnell models. 
        # For standard SD1.5/SDXL, remove it or set it to 7.5
        if model_name == "sd3":
            diffusers_kwargs = {"guidance_scale": 5.0, "num_inference_steps": 28}
        else:
            diffusers_kwargs = {"guidance_scale": 0.0, "num_inference_steps": 2} if "turbo" in MODEL_ID or "schnell" in MODEL_ID else {}
    
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
            
            if is_janus:
                # Janus-Pro-7B generation path
                # NOTE: This implementation is based on typical transformers usage patterns.
                # The exact API may need adjustment based on DeepSeek's Janus documentation.
                # Check https://github.com/deepseek-ai/Janus for the correct usage.
                try:
                    # Set random seed for reproducibility
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)
                    
                    # Process the prompt
                    # Janus may use a different prompt format - adjust as needed
                    inputs = processor(text=prompt, return_tensors="pt")
                    if torch.cuda.is_available():
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    # Generate image tokens
                    # Janus generates images as tokens, which need to be decoded
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=512,  # Adjust based on Janus requirements
                            do_sample=True,
                            temperature=0.7,
                        )
                    
                    # Decode image from tokens
                    # The exact method depends on Janus's processor implementation
                    # Try different approaches based on what the processor supports
                    generated_image = None
                    if hasattr(processor, 'decode_image'):
                        generated_image = processor.decode_image(outputs)
                    elif hasattr(processor, 'batch_decode'):
                        # If outputs are token IDs, decode them
                        decoded = processor.batch_decode(outputs, skip_special_tokens=True)
                        # Janus may return image data in a specific format - adjust as needed
                        # This is a placeholder that may need modification
                        if len(decoded) > 0 and hasattr(decoded[0], 'images'):
                            generated_image = decoded[0].images[0] if decoded[0].images else None
                    else:
                        # Fallback: try to extract image from model outputs directly
                        # This is highly model-specific and may need adjustment
                        if hasattr(outputs, 'images'):
                            generated_image = outputs.images[0] if len(outputs.images) > 0 else None
                        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                            # Try to find image in outputs
                            for item in outputs:
                                if isinstance(item, Image.Image):
                                    generated_image = item
                                    break
                    
                    if generated_image is None:
                        raise ValueError("Could not extract image from Janus output. Check Janus documentation for correct decoding method.")
                    
                    # Ensure it's a PIL Image
                    if not isinstance(generated_image, Image.Image):
                        raise ValueError(f"Expected PIL Image, got {type(generated_image)}")
                    
                    generated_image.save(output_filename)
                    print(f"  Saved to {output_filename}")
                except Exception as e:
                    print(f"  ERROR generating image with Janus: {e}")
                    import traceback
                    print(f"  Traceback: {traceback.format_exc()}")
                    print(f"  Note: Janus-Pro-7B integration may need adjustment based on DeepSeek's API")
                    print(f"  Check: https://github.com/deepseek-ai/Janus for correct usage")
                    continue
            else:
                # Standard diffusers generation path
                generator = torch.Generator(device="cuda").manual_seed(seed)
                
                image = pipe(prompt=prompt,
                             generator=generator,
                             **diffusers_kwargs).images[0]
                
                image.save(output_filename)
                print(f"  Saved to {output_filename}")
    
    # Clean up model to free GPU memory before loading next model
    if is_janus:
        del model
        del processor
    else:
        del pipe
    torch.cuda.empty_cache()
    print(f"Completed model: {model_name}")

print(f"\n{'='*60}")
print("All generations complete!")
print(f"{'='*60}")

