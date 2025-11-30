import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate similarity between original logos and generated images. "
            "Uses multiple visual metrics (LPIPS, CLIP, DINO, DreamSim)."
        )
    )
    parser.add_argument(
        "--original-output-csv",
        type=str,
        default="image_similarity_results.csv",
        help="Output CSV filename for original-to-generated comparisons (will be written in the logos directory).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run models on (e.g., 'cuda', 'cpu'). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--no-lpips",
        action="store_true",
        help="Disable LPIPS metric (if you don't have the lpips package installed).",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Disable CLIP metric.",
    )
    parser.add_argument(
        "--no-dino",
        action="store_true",
        help="Disable DINO metric.",
    )
    parser.add_argument(
        "--no-dreamsim",
        action="store_true",
        help="Disable DreamSim metric.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Limit comparisons to only these model names (e.g., --models sd3 dalle3). If not specified, all models are included.",
    )
    parser.add_argument(
        "--icons",
        type=str,
        nargs="+",
        default=None,
        help="Limit comparisons to only these icon base names (e.g., --icons ios_icon_0 ios_icon_1). If not specified, all icons are included.",
    )
    return parser.parse_args()


def get_device(arg_device: Optional[str]) -> torch.device:
    if arg_device is not None:
        return torch.device(arg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_image_pairs(
    generated_dir: Path,
    allowed_icons: Optional[List[str]] = None,
) -> List[Tuple[str, Path, List[Path]]]:
    """
    For each original logo image saved in generated_images (as *_original.png),
    find all corresponding generated images.
    
    Args:
        generated_dir: Directory containing generated images
        allowed_icons: Optional list of icon base names to include (e.g., ["ios_icon_0"])
    
    Returns list of tuples: (base_name, original_image_path, list_of_generated_paths)
    """
    pairs: List[Tuple[str, Path, List[Path]]] = []
    
    # Find all original images
    original_images = sorted(generated_dir.glob("*_original.png"))
    
    # Convert allowed_icons to set for faster lookup
    allowed_set = set(allowed_icons) if allowed_icons is not None else None
    
    for orig_path in original_images:
        # Extract base name (e.g., "ios_icon_0" from "ios_icon_0_original.png")
        base_name = orig_path.stem.replace("_original", "")
        
        # Filter by icon name if specified
        if allowed_set is not None and base_name not in allowed_set:
            continue
        
        # Find all generated images for this base name (excluding the original)
        gens = sorted([p for p in generated_dir.glob(f"{base_name}_*.png") 
                      if p.name != orig_path.name])
        
        if gens:
            pairs.append((base_name, orig_path, gens))
    
    return pairs


def load_pil_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """Convert PIL image to float tensor in [0, 1], shape (1, 3, H, W)."""
    arr = torch.from_numpy(
        torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        .view(img.size[1], img.size[0], 3)
        .numpy()
    )
    # arr is H x W x 3 uint8
    arr = arr.float() / 255.0
    arr = arr.permute(2, 0, 1).unsqueeze(0).to(device)
    return arr


def resize_to_match(
    img: Image.Image,
    target_size: Tuple[int, int],
) -> Image.Image:
    return img.resize(target_size, Image.BICUBIC)


def maybe_import_lpips():
    try:
        import lpips  # type: ignore
    except Exception:
        return None
    return lpips


def setup_lpips(device: torch.device):
    lpips_module = maybe_import_lpips()
    if lpips_module is None:
        return None
    metric = lpips_module.LPIPS(net="vgg").to(device)
    metric.eval()
    return metric


def compute_lpips_features(
    metric,
    img_tensor: torch.Tensor,
) -> Optional[object]:
    """
    Extract LPIPS features from a single image tensor.
    
    Returns the feature (which may be a list/tuple of tensors from multiple layers)
    that can be used to compute distances. Features are moved to CPU to save GPU memory.
    """
    if metric is None:
        return None
    # LPIPS expects inputs in [-1, 1]
    img_in = (img_tensor * 2.0) - 1.0
    with torch.no_grad():
        # Access the network directly to extract features
        # LPIPS computes features at multiple layers and then computes weighted distance
        # The network returns features from multiple layers (list or tuple of tensors)
        feat = metric.net(img_in)
        # Move features to CPU to save GPU memory
        if isinstance(feat, (list, tuple)):
            feat = [f.cpu() for f in feat]
        else:
            feat = feat.cpu()
    return feat


def compute_lpips_from_features(
    metric,
    feat1: object,
    feat2: object,
    device: torch.device,
) -> Optional[float]:
    """
    Compute LPIPS distance from precomputed features.
    
    LPIPS is a distance metric where lower values indicate greater similarity.
    - Lower is better (0.0 = identical)
    - Higher values indicate more perceptual difference
    
    Features may be a list/tuple of tensors from multiple network layers.
    Features are moved to device for computation, then moved back to CPU.
    """
    if metric is None or feat1 is None or feat2 is None:
        return None
    
    with torch.no_grad():
        # LPIPS computes differences at each layer, then applies learned weights
        # The LPIPS metric has multiple linear layers (lin0, lin1, etc.) for each feature layer
        # We need to compute the distance the same way LPIPS does internally
        if isinstance(feat1, (list, tuple)) and isinstance(feat2, (list, tuple)):
            # Multi-layer features: compute weighted differences at each layer
            # LPIPS uses spatial average pooling and then applies linear layers
            dists = []
            for i, (f1, f2) in enumerate(zip(feat1, feat2)):
                # Move to device for computation
                f1 = f1.to(device)
                f2 = f2.to(device)
                # Normalize features (LPIPS does this internally)
                f1_norm = f1 / (torch.norm(f1, dim=1, keepdim=True) + 1e-10)
                f2_norm = f2 / (torch.norm(f2, dim=1, keepdim=True) + 1e-10)
                # Compute difference
                diff = (f1_norm - f2_norm) ** 2
                # Spatial average pooling
                diff_pooled = torch.mean(diff, dim=(2, 3), keepdim=True)
                # Apply linear layer (lin0, lin1, etc.)
                lin_layer = getattr(metric, f'lin{i}')
                dist = lin_layer(diff_pooled)
                dists.append(dist)
                # Clear GPU memory
                del f1, f2, f1_norm, f2_norm, diff, diff_pooled
            # Sum across layers
            d = sum(dists)
        else:
            # Single tensor - shouldn't happen with standard LPIPS, but handle it
            feat1 = feat1.to(device)
            feat2 = feat2.to(device)
            f1_norm = feat1 / (torch.norm(feat1, dim=1, keepdim=True) + 1e-10)
            f2_norm = feat2 / (torch.norm(feat2, dim=1, keepdim=True) + 1e-10)
            diff = (f1_norm - f2_norm) ** 2
            diff_pooled = torch.mean(diff, dim=(2, 3), keepdim=True)
            d = metric.lin0(diff_pooled)
            del feat1, feat2, f1_norm, f2_norm, diff, diff_pooled
    return float(d.item())


def compute_lpips(
    metric,
    orig: torch.Tensor,
    gen: torch.Tensor,
) -> Optional[float]:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity) distance.
    
    LPIPS is a distance metric where lower values indicate greater similarity.
    - Lower is better (0.0 = identical)
    - Higher values indicate more perceptual difference
    
    Returns the raw LPIPS distance value.
    """
    if metric is None:
        return None
    # LPIPS expects inputs in [-1, 1]
    orig_in = (orig * 2.0) - 1.0
    gen_in = (gen * 2.0) - 1.0
    with torch.no_grad():
        d = metric(orig_in, gen_in)
    return float(d.item())


def setup_clip(device: torch.device):
    from transformers import CLIPModel, CLIPImageProcessor  # type: ignore

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor


def compute_clip_embedding(
    model,
    processor,
    device: torch.device,
    img: Image.Image,
) -> torch.Tensor:
    """Compute CLIP embedding for a single image. Returns CPU tensor to save GPU memory."""
    import torch.nn.functional as F
    
    inputs = processor(
        images=[img],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        feat = model.get_image_features(**inputs)
    feat = F.normalize(feat, dim=-1)
    feat = feat.squeeze(0)  # Remove batch dimension
    return feat.cpu()  # Move to CPU to save GPU memory


def compute_clip_cosine_from_embeddings(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
) -> float:
    """Compute cosine similarity between two CLIP embeddings. Works with CPU tensors."""
    # Embeddings are on CPU, computation can be done there
    sim = torch.sum(emb1 * emb2).item()
    return float(sim)


def compute_clip_cosine(
    model,
    processor,
    device: torch.device,
    orig_img: Image.Image,
    gen_img: Image.Image,
) -> float:
    """Legacy function for backward compatibility."""
    emb1 = compute_clip_embedding(model, processor, device, orig_img)
    emb2 = compute_clip_embedding(model, processor, device, gen_img)
    return compute_clip_cosine_from_embeddings(emb1, emb2)


def setup_dino(device: torch.device):
    # Uses torch.hub; this will download the Dinov2 model code/weights on first run.
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")  # type: ignore
    model.to(device)
    model.eval()

    from torchvision import transforms  # type: ignore

    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return model, transform


def compute_dino_embedding(
    model,
    transform,
    device: torch.device,
    img: Image.Image,
) -> torch.Tensor:
    """Compute DINO embedding for a single image. Returns CPU tensor to save GPU memory."""
    import torch.nn.functional as F
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(img_tensor)
    feat = F.normalize(feat, dim=-1)
    feat = feat.squeeze(0)  # Remove batch dimension
    return feat.cpu()  # Move to CPU to save GPU memory


def compute_dino_cosine_from_embeddings(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
) -> float:
    """Compute cosine similarity between two DINO embeddings. Works with CPU tensors."""
    # Embeddings are on CPU, computation can be done there
    sim = torch.sum(emb1 * emb2).item()
    return float(sim)


def compute_dino_cosine(
    model,
    transform,
    device: torch.device,
    orig_img: Image.Image,
    gen_img: Image.Image,
) -> float:
    """Legacy function for backward compatibility."""
    emb1 = compute_dino_embedding(model, transform, device, orig_img)
    emb2 = compute_dino_embedding(model, transform, device, gen_img)
    return compute_dino_cosine_from_embeddings(emb1, emb2)


def maybe_import_dreamsim():
    try:
        from dreamsim import dreamsim  # type: ignore
    except Exception:
        return None
    return dreamsim


def setup_dreamsim(device: torch.device):
    """Set up DreamSim model and preprocess function."""
    dreamsim_module = maybe_import_dreamsim()
    if dreamsim_module is None:
        return None, None
    
    try:
        # Expand ~ to actual home directory path
        cache_dir = os.path.expanduser("~/.cache")
        model, preprocess = dreamsim_module(pretrained=True, cache_dir=cache_dir)
        model.to(device)
        model.eval()
        return model, preprocess
    except Exception as exc:
        print(f"Error setting up DreamSim: {exc}")
        return None, None


def compute_dreamsim_embedding(
    model,
    preprocess,
    device: torch.device,
    img: Image.Image,
) -> Optional[torch.Tensor]:
    """
    Compute DreamSim embedding for a single image.
    
    Note: DreamSim's API may not directly support embedding extraction.
    This function attempts to extract embeddings if possible, otherwise returns None.
    Returns CPU tensor to save GPU memory.
    """
    if model is None or preprocess is None:
        return None
    
    try:
        # Preprocess image
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Try to extract embeddings from the model
            # DreamSim models typically have a forward method that can return features
            # This is model-specific and may need adjustment based on the actual DreamSim implementation
            if hasattr(model, 'get_image_features'):
                feat = model.get_image_features(img_tensor)
            elif hasattr(model, 'encode_image'):
                feat = model.encode_image(img_tensor)
            else:
                # If we can't extract embeddings directly, return None
                # The direct similarity computation will be used instead
                return None
            
            # Normalize if needed
            import torch.nn.functional as F
            if feat.dim() > 1:
                feat = F.normalize(feat, dim=-1)
            feat = feat.squeeze(0)  # Remove batch dimension
            return feat.cpu()  # Move to CPU to save GPU memory
    except Exception:
        # If embedding extraction fails, return None and use direct computation
        return None


def compute_dreamsim_similarity_from_embeddings(
    emb1: torch.Tensor,
    emb2: torch.Tensor,
) -> float:
    """Compute cosine similarity between two DreamSim embeddings. Works with CPU tensors."""
    # Embeddings are on CPU, computation can be done there
    sim = torch.sum(emb1 * emb2).item()
    return float(sim)


def compute_dreamsim_similarity(
    model,
    preprocess,
    device: torch.device,
    orig_img: Image.Image,
    gen_img: Image.Image,
) -> Optional[float]:
    """
    Compute DreamSim distance between two images.
    
    DreamSim is a distance metric where lower values indicate greater similarity.
    - Lower is better (0.0 = identical)
    - Higher values indicate more perceptual difference
    
    Returns the distance score (typically in [0, 1] range).
    """
    if model is None or preprocess is None:
        return None
    
    try:
        # Preprocess both images (preprocess returns tensors)
        orig_tensor = preprocess(orig_img)
        gen_tensor = preprocess(gen_img)
        
        # Move to device if not already there
        if isinstance(orig_tensor, torch.Tensor):
            orig_tensor = orig_tensor.to(device)
        if isinstance(gen_tensor, torch.Tensor):
            gen_tensor = gen_tensor.to(device)
        
        with torch.no_grad():
            # DreamSim model computes similarity directly when given two images
            similarity = model(orig_tensor, gen_tensor)
            
            # The output might be a tensor or a scalar
            if isinstance(similarity, torch.Tensor):
                similarity = similarity.item()
            
            return float(similarity)
    except Exception as exc:
        print(f"Error computing DreamSim similarity: {exc}")
        return None


def parse_model_and_seed(base: str, generated_name: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Recover (model_name, seed) from a generated filename assuming the
    current convention:

        {base}_{model}_{seed}.png

    Example:
        ios_icon_0_sd3_0.png -> model="sd3", seed=0
    """
    stem = Path(generated_name).stem
    if not stem.startswith(base + "_"):
        return None, None
    rest = stem[len(base) + 1 :]

    m = re.match(r"(.+)_([0-9]+)$", rest)
    if not m:
        return None, None

    model_name = m.group(1)
    seed = int(m.group(2))
    return model_name, seed


def normalize_optional_str(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    value_str = str(value).strip()
    return value_str or None


def normalize_optional_int(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return int(value)
    if isinstance(value, int):
        return value
    value_str = str(value).strip()
    if not value_str:
        return None
    try:
        return int(value_str)
    except ValueError:
        try:
            return int(float(value_str))
        except ValueError:
            return None


def make_result_key_original(
    base_name: str,
    model_name: Optional[object],
    seed: Optional[object],
) -> Tuple[str, Optional[str], Optional[int]]:
    """Create a comparable key for identifying an original-to-generated comparison."""
    base = str(base_name).strip()
    model = normalize_optional_str(model_name)
    normalized_seed = normalize_optional_int(seed)
    return base, model, normalized_seed


def load_existing_results(output_path: Path) -> List[Dict]:
    if not output_path.exists():
        return []
    try:
        df = pd.read_csv(output_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Could not read existing results at {output_path}: {exc}")
        return []
    return df.to_dict("records")


def write_results_csv(results: List[Dict], output_path: Path) -> None:
    if not results:
        print("No results to write.")
        return

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(results)} rows to {output_path}")


def main() -> None:
    args = parse_args()
    logos_dir = Path(__file__).parent
    generated_dir = logos_dir / "generated_images"
    original_output_path = logos_dir / args.original_output_csv

    if not generated_dir.exists():
        raise FileNotFoundError(f"generated_images directory not found at {generated_dir}")

    device = get_device(args.device)
    print(f"Using device: {device}")

    # Set up metrics
    lpips_metric = None
    clip_model = clip_processor = None
    dino_model = dino_transform = None
    dreamsim_model = dreamsim_preprocess = None

    if not args.no_lpips:
        lpips_metric = setup_lpips(device)
        if lpips_metric is None:
            print("LPIPS package not found; LPIPS metric will be skipped.")

    if not args.no_clip:
        clip_model, clip_processor = setup_clip(device)

    if not args.no_dino:
        dino_model, dino_transform = setup_dino(device)

    if not args.no_dreamsim:
        dreamsim_model, dreamsim_preprocess = setup_dreamsim(device)
        if dreamsim_model is None or dreamsim_preprocess is None:
            print("DreamSim package not found or failed to load; DreamSim metric will be skipped.")

    # Filter models if specified
    allowed_models = set(args.models) if args.models is not None else None
    if allowed_models is not None:
        print(f"\nFiltering to models: {sorted(allowed_models)}")

    # Filter icons if specified
    if args.icons is not None:
        print(f"\nFiltering to icons: {sorted(args.icons)}")

    # ===== Original to Generated Comparisons =====
    print("\n" + "="*60)
    print("Computing original-to-generated comparisons")
    print("="*60)
    
    pairs = find_image_pairs(generated_dir, allowed_icons=args.icons)
    print(f"Found {len(pairs)} original logo(s) with at least one generated image.")

    existing_original_records = load_existing_results(original_output_path)
    existing_original_keys = {
        make_result_key_original(
            record.get("base_name"),
            record.get("model_name"),
            record.get("seed"),
        )
        for record in existing_original_records
        if record.get("base_name") is not None
    }

    new_original_results: List[Dict] = []

    for base_name, orig_path, gen_paths in pairs:
        print(f"\nOriginal: {orig_path.name}")
        orig_img = load_pil_image(orig_path)

        for gen_path in gen_paths:
            model_name, seed = parse_model_and_seed(base_name, gen_path.name)
            
            # Skip if model_name is None or if model filtering is enabled and this model is not in the allowed list
            if model_name is None:
                continue
            if allowed_models is not None and model_name not in allowed_models:
                continue
            
            key = make_result_key_original(base_name, model_name, seed)
            if key in existing_original_keys:
                print(f"  Generated: {gen_path.name} (already computed, skipping)")
                continue

            print(f"  Generated: {gen_path.name}")
            gen_img = load_pil_image(gen_path)

            # Resize generated to match original for LPIPS
            if gen_img.size != orig_img.size:
                gen_img_resized = resize_to_match(gen_img, orig_img.size)
            else:
                gen_img_resized = gen_img

            # Prepare tensors for LPIPS
            orig_tensor = to_tensor(orig_img, device)
            gen_tensor = to_tensor(gen_img_resized, device)

            # LPIPS (lower is better - lower values indicate greater similarity)
            lpips_val = compute_lpips(lpips_metric, orig_tensor, gen_tensor) if lpips_metric is not None else None

            # CLIP cosine similarity
            clip_cosine = (
                compute_clip_cosine(clip_model, clip_processor, device, orig_img, gen_img)
                if clip_model is not None and clip_processor is not None
                else None
            )

            # DINO cosine similarity
            dino_cosine = (
                compute_dino_cosine(dino_model, dino_transform, device, orig_img, gen_img)
                if dino_model is not None and dino_transform is not None
                else None
            )

            # DreamSim similarity
            dreamsim_similarity = (
                compute_dreamsim_similarity(dreamsim_model, dreamsim_preprocess, device, orig_img, gen_img)
                if dreamsim_model is not None and dreamsim_preprocess is not None
                else None
            )

            new_original_results.append({
                "base_name": base_name,
                "model_name": model_name,
                "seed": seed,
                "lpips": lpips_val,
                "clip_cosine": clip_cosine,
                "dino_cosine": dino_cosine,
                "dreamsim": dreamsim_similarity,
            })

    combined_original_results = existing_original_records + new_original_results
    write_results_csv(combined_original_results, original_output_path)

    print("\n" + "="*60)
    print("Done. Summary:")
    print("="*60)
    print(f"Original-to-generated comparisons:")
    print(f"  Originals with generated sets: {len(pairs)}")
    print(f"  Newly computed comparisons: {len(new_original_results)}")
    print(f"  Total comparisons written: {len(combined_original_results)}")
    print(f"  Output CSV: {original_output_path}")


if __name__ == "__main__":
    main()



