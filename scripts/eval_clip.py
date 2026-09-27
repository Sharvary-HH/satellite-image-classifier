"""Comparison baseline: zero-shot CLIP on the same eval_set. NOT used by the service.

Usage (from the project root):  python -m scripts.eval_clip

Zero-shot means no training on our tiles at all: CLIP scores how well each image
matches a text prompt per class, and softmax over those scores gives probabilities.
Two prompt sets are compared, to show how much the wording alone matters.

Needs `transformers` (see requirements-clip.txt). First run downloads
openai/clip-vit-base-patch32 (~600 MB) to the Hugging Face cache; later runs are offline.
Writes reports/eval_clip_<prompt_set>.md and .json, same format as scripts/evaluate.py.
"""
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from scripts.evaluate import EVAL_DIR, build_report, load_eval_rows, write_report

MODEL_ID = "openai/clip-vit-base-patch32"
CLASSES = ["AnnualCrop", "Forest", "Highway", "Industrial", "Residential", "River", "SeaLake"]

PROMPT_SETS = {
    # The raw folder names, as-is.
    "raw_names": [f"{c}" for c in CLASSES],
    # Readable descriptions in the style the CLIP paper used for EuroSAT.
    "described": [
        "a centered satellite photo of farmland with annual crops",
        "a centered satellite photo of a forest",
        "a centered satellite photo of a highway or road",
        "a centered satellite photo of an industrial area",
        "a centered satellite photo of a residential area",
        "a centered satellite photo of a river",
        "a centered satellite photo of a sea or lake",
    ],
}


@torch.no_grad()
def main():
    model = CLIPModel.from_pretrained(MODEL_ID).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)  # CLIP's own resize + normalisation

    rows = load_eval_rows()
    images = [Image.open(EVAL_DIR / r["filename"]).convert("RGB") for r in rows]
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"]
    image_emb = model.get_image_features(pixel_values=pixel_values)
    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

    for set_name, prompts in PROMPT_SETS.items():
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
        text_emb = model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # Cosine similarity x CLIP's learned temperature -> logits -> softmax over the 7 classes.
        logits = model.logit_scale.exp() * image_emb @ text_emb.T
        probs = logits.softmax(dim=1).numpy().astype(np.float64)

        report = build_report(rows, probs, CLASSES, {
            "model_name": f"clip-zero-shot ({MODEL_ID})",
            "model_version": f"prompts={set_name}",
            "prompts": dict(zip(CLASSES, prompts)),
        })
        write_report(report, f"eval_clip_{set_name}")


if __name__ == "__main__":
    main()
