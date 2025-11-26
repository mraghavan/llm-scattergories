import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate similarity between original logos (saved as *_original.png in generated_images/) "
            "and generated images in generated_images/ using multiple visual metrics."
        )
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="image_similarity_results.csv",
        help="Output CSV filename (will be written in the logos directory).",
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
    return parser.parse_args()


def get_device(arg_device: Optional[str]) -> torch.device:
    if arg_device is not None:
        return torch.device(arg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_image_pairs(
    generated_dir: Path,
) -> List[Tuple[str, Path, List[Path]]]:
    """
    For each original logo image saved in generated_images (as *_original.png),
    find all corresponding generated images.
    
    Returns list of tuples: (base_name, original_image_path, list_of_generated_paths)
    """
    pairs: List[Tuple[str, Path, List[Path]]] = []
    
    # Find all original images
    original_images = sorted(generated_dir.glob("*_original.png"))
    
    for orig_path in original_images:
        # Extract base name (e.g., "ios_icon_0" from "ios_icon_0_original.png")
        base_name = orig_path.stem.replace("_original", "")
        
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


def compute_lpips(
    metric,
    orig: torch.Tensor,
    gen: torch.Tensor,
) -> Optional[float]:
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


def compute_clip_cosine(
    model,
    processor,
    device: torch.device,
    orig_img: Image.Image,
    gen_img: Image.Image,
) -> float:
    import torch.nn.functional as F

    inputs = processor(
        images=[orig_img, gen_img],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = F.normalize(feats, dim=-1)
    sim = torch.sum(feats[0] * feats[1]).item()
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


def compute_dino_cosine(
    model,
    transform,
    device: torch.device,
    orig_img: Image.Image,
    gen_img: Image.Image,
) -> float:
    import torch.nn.functional as F

    o = transform(orig_img).unsqueeze(0).to(device)
    g = transform(gen_img).unsqueeze(0).to(device)
    with torch.no_grad():
        o_feat = model(o)
        g_feat = model(g)
    o_feat = F.normalize(o_feat, dim=-1)
    g_feat = F.normalize(g_feat, dim=-1)
    sim = torch.sum(o_feat * g_feat).item()
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

    if not generated_dir.exists():
        raise FileNotFoundError(f"generated_images directory not found at {generated_dir}")

    device = get_device(args.device)
    print(f"Using device: {device}")

    pairs = find_image_pairs(generated_dir)
    print(f"Found {len(pairs)} original logo(s) with at least one generated image.")

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

    results: List[Dict] = []

    for base_name, orig_path, gen_paths in pairs:
        print(f"\nOriginal: {orig_path.name}")
        orig_img = load_pil_image(orig_path)

        for gen_path in gen_paths:
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

            # LPIPS
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

            model_name, seed = parse_model_and_seed(base_name, gen_path.name)

            results.append({
                "base_name": base_name,
                "model_name": model_name,
                "seed": seed,
                "lpips": lpips_val,
                "clip_cosine": clip_cosine,
                "dino_cosine": dino_cosine,
            })

    output_path = logos_dir / args.output_csv
    write_results_csv(results, output_path)

    print("\nDone. Summary:")
    print(f"  Originals with generated sets: {len(pairs)}")
    print(f"  Total comparisons: {len(results)}")
    print(f"  Output CSV: {output_path}")


if __name__ == "__main__":
    main()


