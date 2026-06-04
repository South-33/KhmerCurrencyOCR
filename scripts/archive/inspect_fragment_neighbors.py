from __future__ import annotations

import argparse
import csv
from pathlib import Path

from local_runtime import configure_project_cache

configure_project_cache()

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find nearest ImageFolder crops in a fragment classifier embedding.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, help="Source image for a proposal CSV row.")
    parser.add_argument("--csv", required=True, help="Proposal CSV containing x1,y1,x2,y2.")
    parser.add_argument("--row-index", type=int, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-sheet", required=True)
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(class_count: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, class_count)
    return model


def embed(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    features = model.features(images)
    pooled = model.avgpool(features)
    flat = torch.flatten(pooled, 1)
    vector = model.classifier[:-1](flat)
    return torch.nn.functional.normalize(vector, dim=1)


def crop_proposal(image_path: Path, csv_path: Path, row_index: int) -> Image.Image:
    rows = list(csv.DictReader(csv_path.open("r", newline="", encoding="utf-8")))
    if row_index < 0 or row_index >= len(rows):
        raise SystemExit(f"row-index {row_index} outside 0..{len(rows) - 1}")
    row = rows[row_index]
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        x1 = max(0, min(width, float(row["x1"])))
        y1 = max(0, min(height, float(row["y1"])))
        x2 = max(0, min(width, float(row["x2"])))
        y2 = max(0, min(height, float(row["y2"])))
        return image.crop((x1, y1, x2, y2))


def write_sheet(query: Image.Image, rows: list[dict[str, str]], out_path: Path, thumb: int = 180) -> None:
    cols = 5
    label_h = 50
    items = [("QUERY", query, "proposal")] + [
        (f"{row['rank']} {row['class_name']} {row['similarity']}", Image.open(resolve(row["image_path"])).convert("RGB"), row["image_path"])
        for row in rows
    ]
    sheet_h = ((len(items) + cols - 1) // cols) * (thumb + label_h)
    sheet = Image.new("RGB", (cols * thumb, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (title, image, subtitle) in enumerate(items):
        image.thumbnail((thumb, thumb))
        cell_x = (index % cols) * thumb
        cell_y = (index // cols) * (thumb + label_h)
        sheet.paste(image, (cell_x + (thumb - image.width) // 2, cell_y))
        draw.text((cell_x + 4, cell_y + thumb + 2), title[:34], fill=(0, 0, 0))
        draw.text((cell_x + 4, cell_y + thumb + 18), subtitle[:34], fill=(80, 80, 80))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=90)


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(resolve(args.checkpoint), map_location="cpu", weights_only=False)
    class_names: list[str] = checkpoint["classes"]
    image_size = int(checkpoint.get("image_size", 224))
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(resolve(args.data) / args.split, transform=transform, allow_empty=True)
    if dataset.classes != class_names:
        raise SystemExit(f"dataset classes differ from checkpoint: {dataset.classes} != {class_names}")
    device = choose_device(args.device)
    model = build_model(len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    query_image = crop_proposal(resolve(args.image), resolve(args.csv), args.row_index)
    query_tensor = transform(query_image).unsqueeze(0).to(device)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    scored: list[tuple[float, int, str]] = []
    offset = 0
    with torch.no_grad():
        query_embedding = embed(model, query_tensor)
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            sims = (embed(model, images) @ query_embedding.T).squeeze(1).detach().cpu().tolist()
            for batch_index, similarity in enumerate(sims):
                scored.append((float(similarity), int(targets[batch_index]), dataset.samples[offset + batch_index][0]))
            offset += len(targets)
    scored.sort(key=lambda item: item[0], reverse=True)
    rows = [
        {
            "rank": str(rank),
            "similarity": f"{similarity:.6f}",
            "class_name": class_names[target],
            "image_path": str(Path(path).relative_to(ROOT)),
        }
        for rank, (similarity, target, path) in enumerate(scored[: args.top_k], start=1)
    ]
    out_csv = resolve(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "similarity", "class_name", "image_path"])
        writer.writeheader()
        writer.writerows(rows)
    write_sheet(query_image, rows, resolve(args.out_sheet))
    print(f"wrote {len(rows)} neighbors to {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
