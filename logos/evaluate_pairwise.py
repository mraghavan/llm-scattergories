import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate pairwise similarity between generated images for a specific base_name. "
            "Uses multiple visual metrics (LPIPS, CLIP, DINO)."
        )
    )
    parser.add_argument(
        "--base-name",
        type=str,
        required=True,
        help="Base name of the icon to process (e.g., 'ios_icon_0').",
    )
    parser.add_argument(
        "--pairwise-output-csv",
        type=str,
        default="pairwise_similarity_results.csv",
        help="Output CSV filename for pairwise generated image comparisons (will be written in the logos directory).",
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
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Limit comparisons to only these model names (e.g., --models sd3 dalle3). If not specified, all models are included.",
    )
    return parser.parse_args()


def get_device(arg_device: Optional[str]) -> torch.device:
    if arg_device is not None:
        return torch.device(arg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_images_for_base_name(
    generated_dir: Path,
    base_name: str,
) -> List[Tuple[Path, str, int]]:
    """
    Find all generated images for a specific base_name.
    
    Returns list of tuples: (path, model_name, seed)
    """
    # Find all generated images (excluding originals)
    all_images = [p for p in generated_dir.glob("*.png") 
                  if not p.name.endswith("_original.png")]
    
    result: List[Tuple[Path, str, int]] = []
    
    for img_path in all_images:
        stem = img_path.stem
        
        # Try to match this base_name
        if stem.startswith(base_name + "_"):
            model_name, seed = parse_model_and_seed(base_name, img_path.name)
            if model_name is not None and seed is not None:
                result.append((img_path, model_name, seed))
    
    # Sort by model_name, then seed for consistent ordering
    result.sort(key=lambda x: (x[1], x[2]))
    return result


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


def make_result_key_pairwise(
    base_name: str,
    model_name1: Optional[object],
    seed1: Optional[object],
    model_name2: Optional[object],
    seed2: Optional[object],
) -> Tuple[str, Optional[str], Optional[int], Optional[str], Optional[int]]:
    """Create a comparable key for identifying a pairwise generated image comparison."""
    base = str(base_name).strip()
    model1 = normalize_optional_str(model_name1)
    model2 = normalize_optional_str(model_name2)
    norm_seed1 = normalize_optional_int(seed1)
    norm_seed2 = normalize_optional_int(seed2)
    
    # Ensure consistent ordering: model1 <= model2, and if equal, seed1 <= seed2
    if model1 is not None and model2 is not None:
        if model1 > model2:
            model1, model2 = model2, model1
            norm_seed1, norm_seed2 = norm_seed2, norm_seed1
        elif model1 == model2 and norm_seed1 is not None and norm_seed2 is not None:
            if norm_seed1 > norm_seed2:
                norm_seed1, norm_seed2 = norm_seed2, norm_seed1
    
    return base, model1, norm_seed1, model2, norm_seed2


def load_existing_results(output_path: Path) -> List[Dict]:
    if not output_path.exists():
        return []
    try:
        df = pd.read_csv(output_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Could not read existing results at {output_path}: {exc}")
        return []
    return df.to_dict("records")


def append_results_csv(new_results: List[Dict], output_path: Path) -> None:
    """Append new results to existing CSV, or create new file if it doesn't exist."""
    if not new_results:
        print("No new results to write.")
        return

    # Load existing results
    existing_results = load_existing_results(output_path)
    
    # Combine and deduplicate by key
    existing_keys = {
        make_result_key_pairwise(
            record.get("base_name"),
            record.get("model_name1"),
            record.get("seed1"),
            record.get("model_name2"),
            record.get("seed2"),
        )
        for record in existing_results
        if record.get("base_name") is not None
    }
    
    # Add only new results
    combined_results = existing_results.copy()
    for new_record in new_results:
        key = make_result_key_pairwise(
            new_record.get("base_name"),
            new_record.get("model_name1"),
            new_record.get("seed1"),
            new_record.get("model_name2"),
            new_record.get("seed2"),
        )
        if key not in existing_keys:
            combined_results.append(new_record)
            existing_keys.add(key)
    
    # Write combined results
    df = pd.DataFrame(combined_results)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(combined_results)} total rows to {output_path} ({len(new_results)} new)")


def main() -> None:
    args = parse_args()
    logos_dir = Path(__file__).parent
    generated_dir = logos_dir / "generated_images"
    pairwise_output_path = logos_dir / args.pairwise_output_csv

    if not generated_dir.exists():
        raise FileNotFoundError(f"generated_images directory not found at {generated_dir}")

    device = get_device(args.device)
    print(f"Processing base_name: {args.base_name}")
    print(f"Using device: {device}")

    # Set up metrics
    lpips_metric = None
    clip_model = clip_processor = None
    dino_model = dino_transform = None

    if not args.no_lpips:
        lpips_metric = setup_lpips(device)
        if lpips_metric is None:
            print("LPIPS package not found; LPIPS metric will be skipped.")

    if not args.no_clip:
        clip_model, clip_processor = setup_clip(device)

    if not args.no_dino:
        dino_model, dino_transform = setup_dino(device)

    # Filter models if specified
    allowed_models = set(args.models) if args.models is not None else None
    if allowed_models is not None:
        print(f"Filtering to models: {sorted(allowed_models)}")

    # Find images for this base_name
    image_list = find_images_for_base_name(generated_dir, args.base_name)
    
    # Filter by model if specified
    if allowed_models is not None:
        image_list = [(path, model_name, seed) 
                      for path, model_name, seed in image_list 
                      if model_name is not None and model_name in allowed_models]
    
    if len(image_list) < 2:
        print(f"Only {len(image_list)} image(s) found for {args.base_name}. Need at least 2 for pairwise comparison.")
        return
    
    print(f"Found {len(image_list)} images for {args.base_name}")
    
    # Load existing results to skip already computed pairs
    existing_pairwise_records = load_existing_results(pairwise_output_path)
    existing_pairwise_keys = {
        make_result_key_pairwise(
            record.get("base_name"),
            record.get("model_name1"),
            record.get("seed1"),
            record.get("model_name2"),
            record.get("seed2"),
        )
        for record in existing_pairwise_records
        if record.get("base_name") is not None
    }

    # Step 1: Load all images and compute embeddings/features once per image
    print(f"Loading images and computing embeddings...")
    image_data: List[Tuple[Path, str, int, Image.Image, Optional[object], Optional[torch.Tensor], Optional[torch.Tensor]]] = []
    # Structure: (path, model_name, seed, pil_image, lpips_features, clip_embedding, dino_embedding)
    
    # Determine common size for LPIPS (use median size to minimize resizing)
    # Also pre-load images to avoid loading twice
    loaded_images: Dict[Path, Image.Image] = {}
    if lpips_metric is not None and image_list:
        sizes = []
        for path, _, _ in image_list:
            img = load_pil_image(path)
            loaded_images[path] = img
            sizes.append(img.size)
        # Find median size (most common width and height)
        widths = Counter(s for s, _ in sizes)
        heights = Counter(s for _, s in sizes)
        common_width = widths.most_common(1)[0][0]
        common_height = heights.most_common(1)[0][0]
        common_size = (common_width, common_height)
        print(f"Using common size for LPIPS: {common_size}")
    else:
        common_size = None
        # Still need to load images if LPIPS is disabled
        for path, _, _ in image_list:
            if path not in loaded_images:
                loaded_images[path] = load_pil_image(path)
    
    for path, model_name, seed in image_list:
        img = loaded_images[path]
        
        # Compute LPIPS features (resize to common size if needed)
        lpips_features = None
        if lpips_metric is not None:
            if common_size and img.size != common_size:
                img_resized = resize_to_match(img, common_size)
            else:
                img_resized = img
            lpips_tensor = to_tensor(img_resized, device)
            lpips_features = compute_lpips_features(lpips_metric, lpips_tensor)
            # Clear GPU memory after extracting features
            del lpips_tensor
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Compute CLIP embedding
        clip_emb = None
        if clip_model is not None and clip_processor is not None:
            clip_emb = compute_clip_embedding(clip_model, clip_processor, device, img)
            # Clear GPU memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Compute DINO embedding
        dino_emb = None
        if dino_model is not None and dino_transform is not None:
            dino_emb = compute_dino_embedding(dino_model, dino_transform, device, img)
            # Clear GPU memory
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        image_data.append((path, model_name, seed, img, lpips_features, clip_emb, dino_emb))
    
    # Step 2: Compute all pairwise similarities from precomputed embeddings
    print(f"Computing pairwise similarities...")
    new_pairwise_results: List[Dict] = []
    
    for i in range(len(image_data)):
        for j in range(i + 1, len(image_data)):
            path1, model_name1, seed1, img1, lpips_feat1, clip_emb1, dino_emb1 = image_data[i]
            path2, model_name2, seed2, img2, lpips_feat2, clip_emb2, dino_emb2 = image_data[j]
            
            key = make_result_key_pairwise(args.base_name, model_name1, seed1, model_name2, seed2)
            if key in existing_pairwise_keys:
                print(f"  Pair ({model_name1}:{seed1}, {model_name2}:{seed2}): already computed, skipping")
                continue
            
            print(f"  Pair ({model_name1}:{seed1}, {model_name2}:{seed2}): {path1.name} vs {path2.name}")
            
            # LPIPS (lower is better - lower values indicate greater similarity)
            lpips_val = None
            if lpips_metric is not None and lpips_feat1 is not None and lpips_feat2 is not None:
                lpips_val = compute_lpips_from_features(lpips_metric, lpips_feat1, lpips_feat2, device)
            
            # CLIP cosine similarity
            clip_cosine = None
            if clip_emb1 is not None and clip_emb2 is not None:
                clip_cosine = compute_clip_cosine_from_embeddings(clip_emb1, clip_emb2)
            
            # DINO cosine similarity
            dino_cosine = None
            if dino_emb1 is not None and dino_emb2 is not None:
                dino_cosine = compute_dino_cosine_from_embeddings(dino_emb1, dino_emb2)
            
            # Ensure consistent ordering in output (model_name1 <= model_name2)
            if model_name1 is not None and model_name2 is not None:
                if model_name1 > model_name2:
                    model_name1, model_name2 = model_name2, model_name1
                    seed1, seed2 = seed2, seed1
                elif model_name1 == model_name2 and seed1 is not None and seed2 is not None:
                    if seed1 > seed2:
                        seed1, seed2 = seed2, seed1
            
            new_pairwise_results.append({
                "base_name": args.base_name,
                "model_name1": model_name1,
                "seed1": seed1,
                "model_name2": model_name2,
                "seed2": seed2,
                "lpips": lpips_val,
                "clip_cosine": clip_cosine,
                "dino_cosine": dino_cosine,
            })
    
    # Append results to CSV
    append_results_csv(new_pairwise_results, pairwise_output_path)
    
    print(f"\nDone processing {args.base_name}:")
    print(f"  Images processed: {len(image_list)}")
    print(f"  New pairwise comparisons: {len(new_pairwise_results)}")
    print(f"  Output CSV: {pairwise_output_path}")


if __name__ == "__main__":
    main()

