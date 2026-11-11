#!/usr/bin/env python3
"""Infer from a previously trained Keras checkpoint, without remote URL requests."""
import argparse
import json
from pathlib import Path
import tensorflow as tf
from phishcnn.model import probabilities


def main():
    p = argparse.ArgumentParser(description="Research-only phishing classifier")
    p.add_argument("--model", default="artifacts/no_prefix_multi_357_seed42.keras")
    p.add_argument("--url", help="One URL to score")
    p.add_argument("--file", help="Text file containing one URL per line")
    args = p.parse_args()
    if not (args.url or args.file):
        p.error("Provide --url or --file")
    path = Path(args.model)
    if not path.exists():
        p.error(f"Model missing: {path}. Run experiments first.")
    meta = json.loads(path.with_suffix(".json").read_text())
    model = tf.keras.models.load_model(path, compile=False)
    urls = ([args.url] if args.url else [])
    if args.file:
        urls += [s.strip() for s in Path(args.file).read_text().splitlines() if s.strip()]
    probabilities_out = probabilities(model, urls, meta["representation"])
    for url, prob in zip(urls, probabilities_out):
        print(json.dumps({"url": url, "phishing_probability": round(float(prob), 6),
                          "prediction": "phishing" if prob >= meta["threshold"] else "legitimate"}))
    print("Research prototype only; a URL score is not a safety guarantee.")


if __name__ == "__main__":
    main()
