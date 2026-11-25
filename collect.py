"""Stream the published CFPB book extract and save a reproducible sample."""

import argparse
import csv
import gzip
import hashlib
import heapq
import io
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
ARCHIVE = "https://raw.githubusercontent.com/EmilHvitfeldt/smltar/master/data/complaints.csv.gz"
MODEL_URL = (
    "https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/resolve/"
    "30b0a37ccaaa32f332884b96992754e246e48c5f"
)
CLASS_MAP = {
    "Bank account or service": "Bank account",
    "Checking or savings account": "Bank account",
    "Credit card": "Credit card",
    "Credit card or prepaid card": "Credit card",
    "Credit reporting": "Credit reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Debt collection": "Debt collection",
    "Mortgage": "Mortgage",
}


def request(url):
    return urlopen(Request(url, headers={"User-Agent": "Ledger-Capstone/1.0"}), timeout=45)


def collect(output, cap):
    output.mkdir(parents=True, exist_ok=True)
    sampled = defaultdict(list)
    original = Counter()
    eligible = Counter()
    with request(ARCHIVE) as response, gzip.GzipFile(fileobj=response) as compressed:
        etag = response.headers.get("ETag")
        stream = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
        reader = csv.DictReader(stream)
        for row in reader:
            product = row["product"]
            original[product] += 1
            label = CLASS_MAP.get(product)
            sub_product = row["sub_product"].casefold()
            if label == "Credit card" and "prepaid" in sub_product:
                continue
            if label == "Credit reporting" and "credit reporting" not in sub_product:
                continue
            if label is None or not row["consumer_complaint_narrative"].strip():
                continue
            complaint_id = row["complaint_id"]
            eligible[label] += 1
            # A stable priority gives a reproducible sample independent of source order.
            priority = int.from_bytes(hashlib.blake2b(complaint_id.encode(), digest_size=8).digest(), "big")
            record = (priority, complaint_id, row["date_received"], label, row["consumer_complaint_narrative"])
            pool = sampled[label]
            if len(pool) < cap:
                heapq.heappush(pool, record)
            elif record > pool[0]:
                heapq.heapreplace(pool, record)

    output_file = output / "complaints.csv"
    with output_file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "date", "label", "text"))
        for label, records in sorted(sampled.items()):
            writer.writerows(record[1:] for record in sorted(records))
    metadata = {
        "source": ARCHIVE,
        "archive_etag": etag,
        "archive_rows": sum(original.values()),
        "archive_products": dict(original),
        "eligible_by_label": dict(eligible),
        "sampled_by_label": {label: len(records) for label, records in sorted(sampled.items())},
        "first_sample_date": min(record[2] for records in sampled.values() for record in records),
        "last_sample_date": max(record[2] for records in sampled.values() for record in records),
        "sample_sha256": hashlib.sha256(output_file.read_bytes()).hexdigest(),
    }
    (output / "source.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {sum(metadata['sampled_by_label'].values())} sampled narratives: {metadata['sampled_by_label']}", flush=True)


def download_weights():
    directory = ROOT / ".build" / "pretrained"
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "vocab.txt"):
        destination = directory / name
        staging = destination.with_suffix(destination.suffix + ".download")
        with request(f"{MODEL_URL}/{name}") as response, staging.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        staging.replace(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", action="store_true", help="Download the pinned BERT-Tiny weights instead of sampling data.")
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--per-class", type=int, default=620)
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("Set a positive per-class sample limit.")
    if args.weights:
        download_weights()
    else:
        collect(args.output, args.per_class)
