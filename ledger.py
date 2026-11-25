"""Text preparation and a small, CPU-only BERT classifier."""

import json
import os
import re
import struct
import unicodedata
from collections import Counter
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np

ROOT = Path(__file__).resolve().parent
LABELS = ("Bank account", "Credit card", "Credit reporting", "Debt collection", "Mortgage")
MODEL_NAME = "Ledger BERT"
ROUTER_NAME = "Ledger ensemble"
MAX_TOKENS = 128
MAX_TEXT = 40_000
MAX_FILE = 8 * 1024 * 1024
POOL_TOKENS = 32
MIN_WORDS = 8
LAYER_NORM_EPSILON = 1e-12
OUTPUT = "bert.encoder.layer.1.output."


def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\b(?:x{2,}|redacted)\b", " ", text, flags=re.I)
    text = "".join(
        " " if char.isspace() else char
        for char in text
        if char.isspace() or not unicodedata.category(char).startswith("C")
    )
    return " ".join(text.split())


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def terms(text):
    tokens = words(text)
    return Counter(tokens + [first + " " + second for first, second in zip(tokens, tokens[1:])])


class TfidfBayes:
    def __init__(self, vocabulary, idf, log_likelihood, prior):
        self.vocabulary = vocabulary
        self.idf = idf
        self.log_likelihood = log_likelihood
        self.prior = prior

    def vector(self, counts):
        pairs = [(self.vocabulary[term], count) for term, count in counts.items() if term in self.vocabulary]
        ids = np.array([index for index, _ in pairs], dtype=np.int32)
        values = np.log1p([count for _, count in pairs]) * self.idf[ids]
        norm = np.linalg.norm(values)
        return ids, values / norm if norm else values

    def logits(self, text):
        ids, values = self.vector(terms(text))
        return self.log_likelihood[:, ids] @ values + self.prior

    @classmethod
    def fit(cls, texts, labels):
        """Fit the vocabulary, IDF and class likelihoods on training text only."""
        training = [terms(text) for text in texts]
        frequency = Counter()
        for counts in training:
            frequency.update(counts.keys())
        vocabulary = {term: i for i, term in enumerate(sorted(t for t, df in frequency.items() if df >= 2))}
        idf = np.array([np.log((1 + len(training)) / (1 + frequency[term])) + 1 for term in vocabulary])
        prior = np.log(np.bincount(labels, minlength=len(LABELS)) / len(training))
        model = cls(vocabulary, idf, None, prior)
        feature_count = np.full((len(LABELS), len(vocabulary)), 0.1)
        for label, counts in zip(labels, training):
            ids, values = model.vector(counts)
            feature_count[label, ids] += values
        model.log_likelihood = np.log(feature_count / feature_count.sum(axis=1, keepdims=True))
        return model


class WordPiece:
    def __init__(self, path):
        self.vocab = {token: index for index, token in enumerate(path.read_text(encoding="utf-8").splitlines())}
        self.unknown = self.vocab["[UNK]"]

    def _split(self, token):
        if len(token) > 100:
            return [self.unknown]
        pieces = []
        start = 0
        while start < len(token):
            prefix = "##" if start else ""
            for end in range(len(token), start, -1):
                number = self.vocab.get(prefix + token[start:end])
                if number is not None:
                    pieces.append(number)
                    start = end
                    break
            else:
                return [self.unknown]
        return pieces

    def encode(self, text):
        normalized = unicodedata.normalize("NFD", text.lower())
        normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
        tokens = re.findall(r"[^\W_]+|[^\w\s]|_", normalized, flags=re.UNICODE)
        ids = [self.vocab["[CLS]"]]
        for token in tokens:
            ids.extend(self._split(token))
            if len(ids) >= MAX_TOKENS:
                return ids[: MAX_TOKENS - 1] + [self.vocab["[SEP]"]], True
        return ids + [self.vocab["[SEP]"]], False


def load_pretrained(path):
    """Read only the float32 encoder tensors from the pinned safetensors file."""
    with path.open("rb") as source:
        header_length = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_length))
        data = source.read()
    weights = {}
    for name, info in header.items():
        if name.startswith(("bert.embeddings.", "bert.encoder.")):
            if info["dtype"] != "F32":
                raise ValueError(f"Unexpected tensor type for {name}")
            start, end = info["data_offsets"]
            weights[name] = np.frombuffer(data[start:end], dtype="<f4").reshape(info["shape"]).copy()
    if len(weights) != 37:
        raise ValueError(f"Expected 37 BERT encoder tensors, found {len(weights)}")
    return weights


def normalize(x):
    centered = x - x.mean(axis=-1, keepdims=True)
    inverse_std = 1 / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + LAYER_NORM_EPSILON)
    return centered * inverse_std, inverse_std


def softmax(logits):
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def gelu(x):
    # The standard tanh approximation avoids a SciPy dependency.
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class Bert:
    def __init__(self, weights):
        self.weights = weights

    def linear(self, x, name):
        return x @ self.weights[name + ".weight"].T + self.weights[name + ".bias"]

    def layer_norm(self, x, name):
        normalized, _ = normalize(x)
        return normalized * self.weights[name + ".weight"] + self.weights[name + ".bias"]

    def attention(self, x, mask, layer):
        batch, length, width = x.shape
        heads = 2
        depth = width // heads
        query, key, value = (
            self.linear(x, layer + "attention.self." + name)
            .reshape(batch, length, heads, depth)
            .transpose(0, 2, 1, 3)
            for name in ("query", "key", "value")
        )
        scores = (query @ key.swapaxes(-1, -2)) / np.float32(np.sqrt(depth))
        scores = np.where(mask[:, None, None, :], scores, np.float32(-1e4))
        context = (softmax(scores) @ value).transpose(0, 2, 1, 3).reshape(x.shape)
        return self.layer_norm(
            x + self.linear(context, layer + "attention.output.dense"),
            layer + "attention.output.LayerNorm",
        )

    def frozen_features(self, sequences):
        length = max(map(len, sequences))
        ids = np.zeros((len(sequences), length), dtype=np.int32)
        mask = np.zeros(ids.shape, dtype=bool)
        slots = np.zeros((len(sequences), POOL_TOKENS), dtype=np.int32)
        pool_mask = np.zeros_like(slots, dtype=bool)
        for row, sequence in enumerate(sequences):
            size = len(sequence)
            ids[row, :size] = sequence
            mask[row, :size] = True
            # Reserve the first slot for CLS and sample the content evenly.
            count = min(size - 2, POOL_TOKENS - 1)
            slots[row, 1 : count + 1] = np.linspace(1, size - 2, count, dtype=np.int32)
            pool_mask[row, 1 : count + 1] = True

        emb = "bert.embeddings."
        x = self.weights[emb + "word_embeddings.weight"][ids]
        x += self.weights[emb + "position_embeddings.weight"][:length]
        x += self.weights[emb + "token_type_embeddings.weight"][0]
        x = self.layer_norm(x, emb + "LayerNorm")

        first = "bert.encoder.layer.0."
        residual = self.attention(x, mask, first)
        intermediate = gelu(self.linear(residual, first + "intermediate.dense"))
        x = self.layer_norm(residual + self.linear(intermediate, first + "output.dense"), first + "output.LayerNorm")

        last = "bert.encoder.layer.1."
        residual = self.attention(x, mask, last)[np.arange(len(sequences))[:, None], slots]
        intermediate = gelu(self.linear(residual, last + "intermediate.dense"))
        pool_weights = pool_mask.astype(np.float32)
        pool_weights /= np.maximum(pool_weights.sum(axis=1, keepdims=True), 1)
        return intermediate, residual, pool_weights

    def logits(self, features):
        return forward_head(features, self.weights)[0]


TRAINABLE = (
    OUTPUT + "dense.weight",
    OUTPUT + "dense.bias",
    OUTPUT + "LayerNorm.weight",
    OUTPUT + "LayerNorm.bias",
    "classifier.weight",
    "classifier.bias",
)


def forward_head(features, weights):
    intermediate, residual, pool_weights = features
    dense, bias, gamma, beta, classifier, classifier_bias = (weights[name] for name in TRAINABLE)
    unit, inverse_std = normalize(intermediate @ dense.T + bias + residual)
    hidden = unit * gamma + beta
    pooled = np.concatenate((hidden[:, 0], np.einsum("bt,bth->bh", pool_weights, hidden)), axis=1)
    return pooled @ classifier.T + classifier_bias, (pooled, unit, inverse_std)


class Classifier:
    def __init__(self, directory=ROOT / "model"):
        with np.load(directory / "ledger-bert.npz", allow_pickle=False) as archive:
            weights = {name: archive[name].astype(np.float32) for name in archive.files if name != "temperature"}
            self.temperature = float(archive["temperature"].item())
        self.bert = Bert(weights)
        self.tokenizer = WordPiece(directory / "vocab.txt")
        config = json.loads((directory / "ensemble.json").read_text(encoding="utf-8"))
        self.bert_weight = config["bert_weight"]
        self.tfidf_temperature = config["tfidf_temperature"]
        self.ensemble_temperature = config["ensemble_temperature"]
        with np.load(directory / "tfidf.npz", allow_pickle=False) as archive:
            self.baseline = TfidfBayes(
                config["vocabulary"], archive["idf"], archive["log_likelihood"], archive["prior"]
            )

    def summarize(self, text, category):
        sentences = [part for part in re.split(r"(?<=[.!?])\s+", text) if len(words(part)) >= 4]
        # Preserve the original order of the opening and the most class-specific excerpt.
        if not sentences:
            sentences = [text]
        selected = [sentences[0]]
        if len(sentences) > 1:
            others = sentences[1:40]
            lexical = np.array([self.baseline.logits(part) for part in others])
            relevance = lexical[:, category] - np.delete(lexical, category, axis=1).max(axis=1)
            relevant = others[int(relevance.argmax())]
            if relevant != selected[0]:
                selected.append(relevant)
        return [part if len(part) <= 280 else part[:279].rsplit(" ", 1)[0] + "…" for part in selected]

    def predict(self, text):
        cleaned = clean_text(text)
        word_count = len(words(cleaned))
        if word_count < MIN_WORDS:
            raise ValueError(f"Enter a complaint with at least {MIN_WORDS} English words.")
        sequence, truncated = self.tokenizer.encode(cleaned)
        neural = self.bert.logits(self.bert.frozen_features([sequence]))[0] / self.temperature
        lexical = self.baseline.logits(cleaned) / self.tfidf_temperature
        combined = self.bert_weight * neural + (1 - self.bert_weight) * lexical
        scores = softmax(combined / self.ensemble_temperature)
        ranked = np.argsort(scores)[::-1]
        winner, runner_up = ranked[:2]
        return {
            "category": LABELS[winner],
            "confidence": float(scores[winner]),
            "scores": [{"label": LABELS[index], "score": float(scores[index])} for index in ranked],
            "needs_review": bool(scores[winner] < 0.65 or scores[winner] - scores[runner_up] < 0.15),
            "word_count": word_count,
            "analyzed_tokens": len(sequence) - 2,
            "truncated": truncated,
            "summary": self.summarize(cleaned, winner),
        }
