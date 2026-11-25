"""Prepare the corpus, fine-tune Ledger BERT and write held-out evaluation."""

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from ledger import (
    Bert,
    LABELS,
    MAX_TOKENS,
    MODEL_NAME,
    OUTPUT,
    ROOT,
    ROUTER_NAME,
    TRAINABLE,
    TfidfBayes,
    WordPiece,
    clean_text,
    load_pretrained,
    np,
    forward_head,
    softmax,
    words,
)

SEED = 15
SPLITS = ("train", "validation", "test")


def prepare(source, per_class, rng):
    counts = Counter()
    by_signature = defaultdict(list)
    with source.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            counts["source_rows"] += 1
            if row["label"] not in LABELS:
                counts["unsupported_label"] += 1
                continue
            text = clean_text(row["text"] or "")
            tokens = words(text)
            if len(tokens) < 20:
                counts["missing_or_short"] += 1
                continue
            signature = " ".join(tokens)
            by_signature[signature].append(
                {"id": row["id"], "date": row["date"], "label": row["label"], "text": text}
            )

    by_label = defaultdict(list)
    for records in by_signature.values():
        if len({row["label"] for row in records}) > 1:
            counts["conflicting_duplicate"] += len(records)
            continue
        counts["same_label_duplicate"] += len(records) - 1
        by_label[records[0]["label"]].append(records[0])

    rows = []
    for label in LABELS:
        candidates = sorted(by_label[label], key=lambda row: row["id"])
        if len(candidates) < 40:
            raise ValueError(f"{label} needs at least 40 usable examples; found {len(candidates)}")
        subset = [candidates[index] for index in rng.permutation(len(candidates))[:per_class]]
        counts["outside_class_cap"] += len(candidates) - len(subset)
        held_out = max(1, round(0.15 * len(subset)))
        for index, row in enumerate(subset):
            split = "test" if index < held_out else "validation" if index < 2 * held_out else "train"
            rows.append({**row, "split": split})
    return rows, dict(counts)


def metrics(actual, predicted):
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    np.add.at(matrix, (actual, predicted), 1)
    per_class = {}
    for index, label in enumerate(LABELS):
        correct = int(matrix[index, index])
        support = int(matrix[index].sum())
        precision = correct / max(1, int(matrix[:, index].sum()))
        recall = correct / max(1, support)
        per_class[label] = dict(
            precision=precision,
            recall=recall,
            f1=2 * precision * recall / (precision + recall) if precision + recall else 0,
            support=support,
        )
    return {
        "accuracy": float(matrix.trace() / matrix.sum()),
        **{f"macro_{key}": float(np.mean([row[key] for row in per_class.values()]))
           for key in ("precision", "recall", "f1")},
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def gradients(features, actual, weights):
    intermediate, _, pool_weights = features
    logits, (pooled, unit, inverse_std) = forward_head(features, weights)
    probability = softmax(logits)
    probability[np.arange(len(actual)), actual] -= 1
    error = probability / len(actual)
    pooled_error = error @ weights["classifier.weight"]
    cls, mean = np.split(pooled_error, 2, axis=-1)
    hidden_error = mean[:, None] * pool_weights[:, :, None]
    hidden_error[:, 0] += cls
    normalized_error = hidden_error * weights[OUTPUT + "LayerNorm.weight"]
    # Backpropagate through LayerNorm and the final encoder projection.
    before_norm = inverse_std * (
        normalized_error
        - normalized_error.mean(axis=-1, keepdims=True)
        - unit * (normalized_error * unit).mean(axis=-1, keepdims=True)
    )
    return (
        before_norm.reshape(-1, before_norm.shape[-1]).T @ intermediate.reshape(-1, intermediate.shape[-1]),
        before_norm.sum(axis=(0, 1)),
        (hidden_error * unit).sum(axis=(0, 1)),
        hidden_error.sum(axis=(0, 1)),
        error.T @ pooled,
        error.sum(axis=0),
    )


def take(features, indices):
    return tuple(part[indices] for part in features)


def batch_logits(bert, features, indices, batch_size=64):
    return np.concatenate([
        bert.logits(take(features, indices[offset : offset + batch_size]))
        for offset in range(0, len(indices), batch_size)
    ])


def fit(bert, features, actual, train, validation, epochs, rng):
    weights = bert.weights
    width = 2 * weights[OUTPUT + "LayerNorm.bias"].size
    weights["classifier.weight"] = rng.normal(0, 0.02, (len(LABELS), width)).astype(np.float32)
    weights["classifier.bias"] = np.zeros(len(LABELS), dtype=np.float32)
    moments = [(np.zeros_like(weights[key]), np.zeros_like(weights[key])) for key in TRAINABLE]
    initial_dense = weights[TRAINABLE[0]].copy()
    best, best_rank, history, step = None, (-1, -np.inf), [], 0
    batch_size = 32
    for epoch in range(1, epochs + 1):
        shuffled = rng.permutation(train)
        for offset in range(0, len(shuffled), batch_size):
            batch = shuffled[offset : offset + batch_size]
            errors = gradients(take(features, batch), actual[batch], weights)
            step += 1
            for key, gradient, (first, second) in zip(TRAINABLE, errors, moments):
                first *= 0.9
                first += 0.1 * gradient
                second *= 0.999
                second += 0.001 * gradient**2
                rate = 0.001 if key.startswith("classifier") else 0.0003
                weights[key] -= rate * (first / (1 - 0.9**step)) / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)

        logits = batch_logits(bert, features, validation)
        probability = softmax(logits)
        expected = actual[validation]
        loss = float(-np.log(np.maximum(probability[np.arange(len(expected)), expected], 1e-8)).mean())
        f1 = metrics(expected, logits.argmax(axis=1))["macro_f1"]
        history.append({"epoch": epoch, "validation_f1": f1, "validation_loss": loss})
        print(f"Epoch {epoch:2d} | validation macro F1 {f1:.4f} | loss {loss:.4f}", flush=True)
        if (f1, -loss) > best_rank:
            best = {key: weights[key].copy() for key in TRAINABLE}
            best_rank = (f1, -loss)
            best_epoch = epoch
    weights.update(best)
    drift = float(np.linalg.norm(weights[TRAINABLE[0]] - initial_dense))
    return best_epoch, history, drift


def calibrate(logits, actual):
    candidates = np.linspace(0.35, 2.5, 87)

    def nll(temperature):
        scores = softmax(logits / temperature)
        return -np.log(np.maximum(scores[np.arange(len(actual)), actual], 1e-8)).mean()

    return float(min(candidates, key=nll))


def fit_ensemble(bert_logits, tfidf_logits, labels):
    bert_temperature = calibrate(bert_logits, labels)
    tfidf_temperature = calibrate(tfidf_logits, labels)
    neural = bert_logits / bert_temperature
    lexical = tfidf_logits / tfidf_temperature

    # Pick the mix on validation only. Leave both predictors in the ensemble.
    candidates = np.arange(0.05, 1, 0.05)

    def validation_f1(weight):
        combined = weight * neural + (1 - weight) * lexical
        return metrics(labels, combined.argmax(axis=1))["macro_f1"]

    weight = float(max(candidates, key=validation_f1))
    ensemble_temperature = calibrate(weight * neural + (1 - weight) * lexical, labels)
    return bert_temperature, tfidf_temperature, weight, ensemble_temperature


def run(source, pretrained, output, per_class, epochs):
    rng = np.random.default_rng(SEED)
    rows, cleaning = prepare(source, per_class, rng)
    tokenizer = WordPiece(pretrained / "vocab.txt")
    encoded = [tokenizer.encode(row["text"]) for row in rows]
    sequences = [ids for ids, _ in encoded]
    actual = np.array([LABELS.index(row["label"]) for row in rows])
    train, validation, test = splits = tuple(
        np.array([index for index, row in enumerate(rows) if row["split"] == split])
        for split in SPLITS
    )
    print(f"{len(rows)} complaints | splits: {[len(part) for part in splits]}", flush=True)
    bert = Bert(load_pretrained(pretrained / "model.safetensors"))
    blocks = []
    for start in range(0, len(rows), 16):
        blocks.append(bert.frozen_features(sequences[start : start + 16]))
        if start % 256 == 0:
            print(f"Encoded {min(start + 16, len(rows))}/{len(rows)}", flush=True)
    features = tuple(np.concatenate(parts) for parts in zip(*blocks))
    del blocks

    epoch, history, drift = fit(bert, features, actual, train, validation, epochs, rng)
    baseline = TfidfBayes.fit([rows[i]["text"] for i in train], actual[train])
    temperature, tfidf_temperature, bert_weight, ensemble_temperature = fit_ensemble(
        batch_logits(bert, features, validation),
        np.array([baseline.logits(rows[i]["text"]) for i in validation]),
        actual[validation],
    )
    bert_logits = batch_logits(bert, features, test)
    tfidf_logits = np.array([baseline.logits(rows[i]["text"]) for i in test])
    combined = bert_weight * bert_logits / temperature + (1 - bert_weight) * tfidf_logits / tfidf_temperature
    predicted = bert_logits.argmax(axis=1)
    lexical = tfidf_logits.argmax(axis=1)
    ensemble = combined.argmax(axis=1)
    report = {
        "model": MODEL_NAME,
        "router": ROUTER_NAME,
        "seed": SEED,
        "labels": LABELS,
        "max_tokens": MAX_TOKENS,
        "data": {
            "cleaning": cleaning,
            "total": len(rows),
            "splits": dict(zip(SPLITS, map(len, splits))),
            "by_class": {
                label: {
                    **dict(Counter(row["split"] for row in rows if row["label"] == label)),
                    "median_words": float(np.median([len(words(row["text"])) for row in rows if row["label"] == label])),
                }
                for label in LABELS
            },
            "truncated": sum(truncated for _, truncated in encoded),
        },
        "training": {
            "selected_epoch": epoch,
            "temperature": temperature,
            "tfidf_temperature": tfidf_temperature,
            "bert_weight": bert_weight,
            "ensemble_temperature": ensemble_temperature,
            "encoder_projection_delta_l2": drift,
            "epochs": history,
        },
        "bert": metrics(actual[test], predicted),
        "baseline": metrics(actual[test], lexical),
        "ensemble": metrics(actual[test], ensemble),
    }
    for directory in ("analysis", "data", "model"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    with (output / "data" / "splits.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "date", "label", "text", "split"])
        writer.writeheader()
        writer.writerows(rows)
    with (output / "analysis" / "test_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("id", "actual", "bert", "baseline", "ensemble"))
        writer.writerows(
            (rows[i]["id"], LABELS[actual[i]], LABELS[p], LABELS[b], LABELS[e])
            for i, p, b, e in zip(test, predicted, lexical, ensemble)
        )
    (output / "analysis" / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "model" / "ledger-bert.npz",
        **bert.weights,
        temperature=np.array(temperature, dtype=np.float32),
    )
    config = {
        "bert_weight": bert_weight,
        "tfidf_temperature": tfidf_temperature,
        "ensemble_temperature": ensemble_temperature,
        "vocabulary": baseline.vocabulary,
    }
    (output / "model" / "ensemble.json").write_text(json.dumps(config) + "\n", encoding="utf-8")
    np.savez_compressed(
        output / "model" / "tfidf.npz",
        idf=baseline.idf,
        log_likelihood=baseline.log_likelihood,
        prior=baseline.prior,
    )
    shutil.copyfile(pretrained / "vocab.txt", output / "model" / "vocab.txt")
    scores = " | ".join(f"{name} {report[name]['macro_f1']:.4f}" for name in ("bert", "baseline", "ensemble"))
    print(f"Test macro F1 | {scores} | ensemble BERT weight {bert_weight:.2f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "complaints.csv")
    parser.add_argument("--pretrained", type=Path, default=ROOT / ".build" / "pretrained")
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--per-class", type=int, default=450)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args()
    if args.per_class < 40 or args.epochs < 1:
        parser.error("Use at least 40 records per class and at least one epoch.")
    run(args.source, args.pretrained, args.output, args.per_class, args.epochs)
