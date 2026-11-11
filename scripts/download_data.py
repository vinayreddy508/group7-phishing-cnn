#!/usr/bin/env python3
"""Download official UCI dataset #967, saving URL + original Label only."""
import argparse
from phishcnn.data import fetch_uci


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/phiusiil_urls.csv")
    args = p.parse_args()
    print("Fetching PhiUSIIL dataset from UCI #967...")
    print(f"Saved to {fetch_uci(args.out)}")
    print("Dataset stays local (ignored by .gitignore).")


if __name__ == "__main__":
    main()
