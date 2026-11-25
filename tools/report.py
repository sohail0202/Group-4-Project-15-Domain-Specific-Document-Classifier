"""Build the print report and analysis notebook from the measured run."""

import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / ".build"
METRICS = json.loads((ROOT / "analysis" / "metrics.json").read_text(encoding="utf-8"))
SOURCE = json.loads((ROOT / "data" / "source.json").read_text(encoding="utf-8"))
DATA = METRICS["data"]
TRAINING = METRICS["training"]
LABELS = METRICS["labels"]
KEYS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
DISPLAY = ("Accuracy", "Macro precision", "Macro recall", "Macro F1")
SYSTEMS = (("Ledger BERT", "bert"), ("TF-IDF + Naive Bayes", "baseline"), ("Ledger ensemble", "ensemble"))
TEAM = (
    ("Drishti Jain", "2401201036"), ("Neha", "2401201037"), ("Yash Lamba", "2401201038"),
    ("Teesha", "2401201039"), ("Sohail Khan", "2401201040"), ("Parth Behera", "2401201041"),
)
ABBR = ["Bank", "Card", "Report", "Debt", "Mortgage"]


def percent(number):
    return f"{number:.1%}"


def table(headers, rows):
    header = "".join(f"<th scope='col'>{escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def bar_chart(items, maximum, fmt):
    return "".join(
        f"<div class='bar-row'><span>{escape(label)}</span>"
        f"<span class='track'><i style='width:{100 * value / maximum:.1f}%'></i></span>"
        f"<span>{fmt(value)}</span></div>"
        for label, value in items
    )


def confusion(matrix):
    head = "<th>Actual / predicted</th>" + "".join(f"<th>{name}</th>" for name in ABBR)
    maximum = max(max(row) for row in matrix)
    cells = []
    for label, row in zip(LABELS, matrix):
        data = "".join(
            f"<td style='background:rgba(57,96,64,{value / maximum * .82:.3f});"
            f"color:{'#fff' if value / maximum > .63 else '#252b22'}'>{value}</td>"
            for value in row
        )
        cells.append(f"<tr><th>{label}</th>{data}</tr>")
    return f"<table class='matrix'><thead><tr>{head}</tr></thead><tbody>{''.join(cells)}</tbody></table>"


def page(number, heading, content, *, cover=False):
    headline = "" if cover else f"<h2>{heading}</h2>"
    return f"""
    <section class="page{' cover' if cover else ''}">
      <header><span>Ledger</span><span>PROJECT 15 / GROUP 4</span></header>
      {headline}
      {content}
      <footer><span>Domain-Specific Document Classifier</span><span>{number:02d} / 05</span></footer>
    </section>"""


def write_report():
    held_out = DATA["splits"]["test"]
    ensemble = METRICS["ensemble"]
    clean = DATA["cleaning"]
    performance = table(
        ("Model", *DISPLAY),
        [(name, *(percent(METRICS[key][k]) for k in KEYS)) for name, key in SYSTEMS],
    )
    class_rows = [
        (label, percent(row["precision"]), percent(row["recall"]), percent(row["f1"]), row["support"])
        for label, row in ensemble["per_class"].items()
    ]
    worst_errors = sorted(
        ((count, LABELS[i], LABELS[j]) for i, row in enumerate(ensemble["confusion_matrix"])
         for j, count in enumerate(row) if i != j),
        reverse=True,
    )[:2]
    errors = " and ".join(f"{count} {actual.lower()} cases labeled {predicted.lower()}" for count, actual, predicted in worst_errors)
    pages = [
        page(1, "", f"""
          <p class="eyebrow">NLP CAPSTONE REPORT</p>
          <h1>Ledger</h1>
          <p class="deck">A document classifier for<br>consumer finance complaints</p>
          <p class="lead">Route an English complaint to one of five product categories so it can reach the right review queue.</p>
          <div class="kpis"><div><strong>{DATA['total']:,}</strong><span>cleaned narratives</span></div>
          <div><strong>{held_out}</strong><span>held-out complaints</span></div>
          <div><strong>{percent(ensemble['macro_f1'])}</strong><span>ensemble macro F1</span></div></div>
          <h3>Team</h3>{table(("Student", "Roll number"), TEAM)}
          <p class="small">Models used in the study: Ledger BERT, TF-IDF with multinomial Naive Bayes, and the Ledger ensemble used by the app.</p>
        """, cover=True),
        page(2, "Corpus and exploratory analysis", f"""
          <p class="lead">We sampled narratives from the CFPB archive distributed with <em>Supervised Machine Learning for Text Analysis in R</em>. The archive contained {SOURCE['archive_rows']:,} public complaint records.</p>
          <div class="duo"><section><h3>Eligible source records</h3>
          {bar_chart([(label, SOURCE['eligible_by_label'][label]) for label in LABELS], max(SOURCE['eligible_by_label'].values()), lambda value: f'{value:,}')}
          <p class="small">Product labels reflect the categories in the published complaints. We excluded prepaid cards and non-credit-reporting subproducts from their broader source labels.</p></section>
          <section><h3>Median words after cleaning</h3>
          {bar_chart([(label, DATA['by_class'][label]['median_words']) for label in LABELS], max(row['median_words'] for row in DATA['by_class'].values()), lambda value: f'{value:g}')}
          <p class="small">{DATA['truncated']:,} of {DATA['total']:,} cleaned narratives exceeded BERT's token budget.</p></section></div>
          <h3>Cleaning and split</h3>
          <p>We drew {clean['source_rows']:,} records with stable sampling based on complaint IDs. The sample spans {SOURCE['first_sample_date']} to {SOURCE['last_sample_date']}. We normalized Unicode and whitespace, removed redaction markers, and dropped records with fewer than 20 words.</p>
          <p>Cleaning removed {clean.get('missing_or_short', 0)} short or empty texts, {clean.get('same_label_duplicate', 0)} repeated same-label narratives, and {clean.get('conflicting_duplicate', 0)} rows with contradictory duplicate labels. We capped each class at 450 and split by label using seed 15.</p>
          {table(("Category", "Train", "Validation", "Test"), [(label, *(DATA['by_class'][label][split] for split in ('train','validation','test'))) for label in LABELS] + [("Total", *(DATA['splits'][split] for split in ('train','validation','test')))])}
          <p class="small">A balanced test set gives each product equal weight. The archive has many more credit-reporting complaints than bank-account complaints.</p>
        """),
        page(3, "Features and model training", f"""
          <p class="lead">We trained two classifiers on the same text. One learns from words and short phrases. The other starts from a pretrained language model.</p>
          <h3>TF-IDF baseline</h3>
          <p>Lowercase word unigrams and bigrams form sparse features. The training set determines the vocabulary and inverse-document-frequency weights. Log-scaled term counts and L2 normalization reduce the effect of document length. A smoothed multinomial Naive Bayes model scores the five classes.</p>
          <h3>Ledger BERT</h3>
          <p>Google's uncased BERT-Tiny has two transformer blocks, two attention heads per block and a 128-wide hidden state. WordPiece turns the first 126 content tokens into IDs. CLS and SEP fill the other two positions.</p>
          <p>We keep embeddings and attention frozen. We backpropagate cross-entropy through the new classifier, the final feed-forward output projection and its LayerNorm. The head uses the CLS state and the mean of up to 31 evenly sampled contextual content states.</p>
          <p>Adam uses a batch size of 32 and learning rates of 0.0003 for the encoder parameters and 0.001 for the head. The final projection changed by an L2 norm of {TRAINING['encoder_projection_delta_l2']:.3f}.</p>
          <h3>Validation and ensemble</h3>
          <p>Epoch {TRAINING['selected_epoch']} had the highest validation macro F1 across {len(TRAINING['epochs'])} epochs. Validation data also set score temperatures and the mixing weight. The app combines {TRAINING['bert_weight']:.0%} BERT and {1-TRAINING['bert_weight']:.0%} TF-IDF logits after temperature scaling.</p>
          {table(("Epoch", "Validation macro F1", "Validation loss"), [(row['epoch'], percent(row['validation_f1']), f"{row['validation_loss']:.3f}") for row in TRAINING['epochs'] if row['epoch'] in {1, 4, 8, TRAINING['selected_epoch'], 12}])}
          <p class="small">The test set did not set the vocabulary, fit weights, select an epoch or tune the ensemble.</p>
        """),
        page(4, "Held-out results", f"""
          <p class="lead">All three models predicted the same {held_out} complaints. Macro scores average the five per-category scores equally.</p>
          {performance}
          <h3>Ledger ensemble by class</h3>
          {table(("Category", "Precision", "Recall", "F1", "Support"), class_rows)}
          <h3>Confusion matrix</h3>
          {confusion(ensemble['confusion_matrix'])}
          <p>The two most frequent error directions were {errors}.</p>
          <p class="small">The comparison includes a full-text TF-IDF branch and a BERT branch limited to the opening tokens. It measures this configuration on a single balanced historical split.</p>
        """),
        page(5, "Application and live demo", """
          <p class="lead">Open Ledger locally. Paste a complaint, load a built-in sample, or upload a TXT, text-based PDF or DOCX.</p>
          <img class="screenshot" src="../screenshots/classification.png" alt="Ledger displaying a credit card complaint and ranked category scores">
          <h3>In class</h3>
          <ol><li>Run <code>python app.py</code> and open <code>127.0.0.1:8765</code>.</li>
          <li>Choose <strong>Card charge</strong>, classify it, then open the extractive summary.</li>
          <li>Switch to <strong>Upload file</strong> and classify <code>examples/mortgage-escrow.pdf</code>.</li></ol>
          <h3>Limitations and next experiment</h3>
          <p>The model assigns one of five labels even if an input falls outside those categories. A low or close score triggers a review cue. These scores can still be wrong for new products, mixed complaints, other countries or later data. Scanned PDFs need OCR.</p>
          <p>Evaluate a later time period and include an out-of-scope set before using this for intake. Measure the trade-off between review volume and missed errors.</p>
          <p class="small sources"><a href="https://www.consumerfinance.gov/data-research/consumer-complaints/">CFPB Complaint Database</a> · <a href="https://smltar.com/appendixdata">Hvitfeldt and Silge data appendix</a> · <a href="https://huggingface.co/google/bert_uncased_L-2_H-128_A-2">Google BERT-Tiny</a> · <a href="https://arxiv.org/abs/1908.08962">Turc et al., 2019</a></p>
        """),
    ]

    css = """
      @page { size: A4; margin: 0; }
      * { box-sizing: border-box; }
      body { margin: 0; font: 11pt/1.49 'Trebuchet MS', sans-serif; color: #252b22; background: #f5f2e8; }
      .page { width: 210mm; height: 296.8mm; padding: 16mm 19mm; background: #fcfaf4; position: relative; break-after: page; }
      .page:last-child { break-after: auto; }
      header, footer { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #dad5c7; padding-bottom: 2.7mm; color: #4e584c; font: 9pt Consolas, monospace; }
      header span:first-child { font: bold 16pt 'Palatino Linotype', serif; color: #304536; }
      footer { border: 0; border-top: 1px solid #dad5c7; padding: 2.5mm 0 0; position: absolute; inset: auto 19mm 12mm; }
      h1, h2, h3, .deck, .kpis strong { font-family: 'Palatino Linotype', Georgia, serif; color: #24362a; }
      h1 { font-size: 75pt; font-weight: normal; line-height: .9; margin: 2mm 0 6mm; letter-spacing: -.07em; }
      h2 { font-size: 32pt; font-weight: normal; line-height: 1.05; margin: 10mm 0 4mm; letter-spacing: -.05em; }
      h3 { font-size: 15pt; line-height: 1.25; margin: 5mm 0 2mm; }
      .deck { font-size: 28pt; line-height: 1.06; margin: 0 0 8mm; letter-spacing: -.035em; }
      .eyebrow { color: #853d2d; letter-spacing: .13em; margin: 14mm 0 3mm; font: bold 10pt Consolas, monospace; }
      .lead { font-size: 12pt; line-height: 1.43; margin: 0 0 6mm; }
      p { margin: 0 0 3mm; }
      .small { font-size: 9pt; color: #50594b; margin-top: 3mm; }
      .kpis { display: grid; grid-template-columns: repeat(3, 1fr); border-block: 1px solid #dad5c7; margin: 8mm 0; padding: 3mm 0 4mm; }
      .kpis strong { display: block; font-size: 30pt; font-weight: normal; letter-spacing: -.06em; line-height: 1.1; }
      .kpis span { color: #50594b; font-size: 9pt; }
      table { width: 100%; border-collapse: collapse; font-size: 9.6pt; margin: 0 0 4mm; font-variant-numeric: tabular-nums; }
      th, td { padding: 2mm 2.2mm; border-bottom: 1px solid #dad5c7; text-align: right; }
      th { color: #304536; background: #efeee3; }
      tr > :first-child { text-align: left; }
      .cover table td { text-align: left; }
      .cover table th { text-align: left; }
      .duo { display: grid; grid-template-columns: 1fr 1fr; gap: 6mm; margin-bottom: 3mm; }
      .bar-row { display: grid; grid-template-columns: 33mm 1fr 14mm; align-items: center; gap: 2mm; border-bottom: 1px solid #e5e1d5; padding: 1.3mm 0; font: 8.4pt 'Trebuchet MS', sans-serif; }
      .bar-row span:last-child { text-align: right; font-variant-numeric: tabular-nums; }
      .track { background: #eae6d9; height: 2mm; }
      .track i { display: block; height: 100%; background: #42664a; }
      .matrix td { text-align: center; }
      .matrix th:not(:first-child) { text-align: center; }
      .screenshot { width: 100%; border: 1px solid #dad5c7; display: block; }
      ol { margin: 0 0 3mm; padding-left: 5mm; }
      li { padding-bottom: 1mm; }
      code { font-family: Consolas, monospace; font-size: 9.5pt; }
      a { color: #304536; }
    """
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8">
    <title>Ledger | Project 15 report</title><style>{css}</style><body>{''.join(pages)}</body></html>"""
    (STAGING / "report.html").write_text(html, encoding="utf-8")


def write_notebook():
    def markdown(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}

    def code(text):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": text.strip().splitlines(keepends=True)}

    cells = [
        markdown("""# Ledger | Project 15
Group 4 · Domain-Specific Document Classifier

Run this notebook from the project directory.
It examines the shipped data splits, reproduces the evaluation from saved predictions, and runs the deployed classifier.

[CFPB source](https://www.consumerfinance.gov/data-research/consumer-complaints/) · [Book extract](https://smltar.com/appendixdata) · [BERT-Tiny](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2)"""),
        code("""import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median

from ledger import Classifier, LABELS, words
from train import metrics

def load_csv(path):
    with open(path, encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))

rows = load_csv('data/splits.csv')
report = json.loads(Path('analysis/metrics.json').read_text(encoding='utf-8'))
print(f'Cleaned complaints: {len(rows):,}')
assert len({row['id'] for row in rows}) == len(rows)
assert len({' '.join(words(row['text'])) for row in rows}) == len(rows)"""),
        markdown("""## Class balance and length
The published archive is imbalanced.
This experiment limits each class to 450 cleaned complaints and stratifies the split.
The table compares the original eligible counts with split counts and shows median length after cleaning."""),
        code("""source = json.loads(Path('data/source.json').read_text(encoding='utf-8'))
counts = Counter((row['label'], row['split']) for row in rows)
print(f"{'Category':20} {'Available':>10} {'Train':>8} {'Val':>8} {'Test':>8} {'Median words':>13}")
for label in LABELS:
    sizes = [counts[label, split] for split in ('train', 'validation', 'test')]
    length = median(len(words(row['text'])) for row in rows if row['label'] == label)
    print(f"{label:20} {source['eligible_by_label'][label]:>10,} {sizes[0]:>8} {sizes[1]:>8} {sizes[2]:>8} {length:>13.0f}")
print('Cleaning audit:', report['data']['cleaning'])
print('BERT-truncated complaints:', report['data']['truncated'])"""),
        markdown("""## Feature engineering and training
The lexical baseline uses normalized TF-IDF word unigrams and bigrams with multinomial Naive Bayes.
The transformer uses WordPiece IDs and updates its final output projection and LayerNorm with a new classifier.
The ensemble mixes temperature-scaled logits and chooses the mix on validation data.

To rerun training, use `python collect.py --weights` and `python train.py --output .build/reproduction` in the project directory."""),
        code("""training = report['training']
print('Selected epoch:', training['selected_epoch'])
print('Final encoder projection L2 change:', round(training['encoder_projection_delta_l2'], 4))
print('BERT weight:', f"{training['bert_weight']:.0%}")
for item in training['epochs']:
    print(f"Epoch {item['epoch']:2d}  val macro F1 {item['validation_f1']:.3f}")"""),
        markdown("""## Held-out evaluation
The test set was held out during training and calibration.
Check accuracy, macro precision, macro recall, and macro F1, followed by the ensemble confusion matrix."""),
        code("""import numpy as np

predictions = load_csv('analysis/test_predictions.csv')
actual = np.array([LABELS.index(row['actual']) for row in predictions])
print(f"{'Model':13} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'Macro F1':>9}")
for key in ('bert', 'baseline', 'ensemble'):
    predicted = np.array([LABELS.index(row[key]) for row in predictions])
    current = metrics(actual, predicted)
    assert current == report[key]
    print(f"{key:13} {current['accuracy']:>9.1%} {current['macro_precision']:>10.1%} "
          f"{current['macro_recall']:>8.1%} {current['macro_f1']:>9.1%}")
print('Rows are true labels, columns are predicted labels:', LABELS)
print(np.array(report['ensemble']['confusion_matrix']))"""),
        markdown("""## Inference
This is the same classifier the web app calls.
The examples are fictional.
BERT reads up to 126 content tokens, while TF-IDF uses all extracted text."""),
        code("""classifier = Classifier()
for example in sorted(Path('examples').glob('*.txt')):
    result = classifier.predict(example.read_text(encoding='utf-8'))
    print(f"{example.name}: {result['category']} ({result['confidence']:.1%})")"""),
    ]
    for index, cell in enumerate(cells, 1):
        cell["id"] = f"ledger-{index:02d}"
    notebook = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = ROOT / "Project_15_Analysis.ipynb"
    target.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    STAGING.mkdir(exist_ok=True)
    write_report()
    write_notebook()
