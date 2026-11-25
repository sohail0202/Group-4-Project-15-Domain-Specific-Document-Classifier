import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const work = path.join(root, ".build");
const output = path.resolve(root, process.env.LEDGER_OUTPUT_DIR ?? "deliverables");
const runtime = process.env.LEDGER_NODE_MODULES;
const skill = process.env.LEDGER_PRESENTATION_SKILL;
const python = process.env.LEDGER_PYTHON ?? "python";
if (!runtime || !skill) throw new Error("Set LEDGER_NODE_MODULES and LEDGER_PRESENTATION_SKILL to the bundled locations.");
process.env.RUNTIME_NODE_MODULES ??= runtime;

const { Presentation, PresentationFile } = await import(pathToFileURL(
  path.join(runtime, "@oai", "artifact-tool", "dist", "artifact_tool.mjs"),
));
const { finalizePresentation } = await import(pathToFileURL(
  path.join(skill, "container_tools", "artifact_tool_utils.mjs"),
));
const report = JSON.parse(await fs.readFile(path.join(root, "analysis", "metrics.json"), "utf8"));
const source = JSON.parse(await fs.readFile(path.join(root, "data", "source.json"), "utf8"));
const { data, labels, training } = report;
const W = 1280;
const H = 720;
const pres = Presentation.create({ slideSize: { width: W, height: H } });
const C = {
  paper: "#F5F2E8", sheet: "#FCFAF4", ink: "#222820", muted: "#53594E",
  rule: "#D8D4C6", green: "#304536", dark: "#202D26", soft: "#D0D7C9",
  forestRule: "#526257", lime: "#DBEB96", warm: "#973E2A", greenTint: "#E2EACF",
};
const serif = "Palatino Linotype";
const body = "Trebuchet MS";
const mono = "Consolas";
const noLine = { fill: "none", width: 0, style: "solid" };
const percent = value => `${(value * 100).toFixed(1)}%`;
const integer = value => value.toLocaleString("en-US");
const team = [
  ["Drishti Jain", "2401201036"], ["Neha", "2401201037"], ["Yash Lamba", "2401201038"],
  ["Teesha", "2401201039"], ["Sohail Khan", "2401201040"], ["Parth Behera", "2401201041"],
];
const dataSource = "CFPB data source: https://www.consumerfinance.gov/data-research/consumer-complaints/\nArchive documentation: https://smltar.com/appendixdata";
const modelSource = "Google BERT-Tiny: https://huggingface.co/google/bert_uncased_L-2_H-128_A-2\nTurc et al. (2019): https://arxiv.org/abs/1908.08962";

function text(slide, value, x, y, width, height, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox", name: options.name ?? value.slice(0, 45),
    position: { left: x, top: y, width, height }, fill: "none", line: noLine,
  });
  shape.text = value;
  shape.text.style = {
    typeface: options.font ?? body, fontSize: options.size ?? 24, color: options.color ?? C.ink,
    bold: options.bold ?? false, italic: options.italic ?? false, alignment: options.align ?? "left",
    verticalAlignment: "middle", wrap: options.wrap ? "square" : "none", autoFit: "none", insets: 0,
  };
  return shape;
}

function rule(slide, x, y, width, fill = C.rule) {
  slide.shapes.add({
    name: "Divider", geometry: "rect", position: { left: x, top: y, width, height: 1.5 }, fill, line: noLine,
  });
}

function slide(dark = false) {
  const page = pres.slides.add();
  const n = pres.slides.items.length;
  const color = dark ? C.soft : C.muted;
  page.background.fill = dark ? C.dark : C.paper;
  rule(page, 72, 650, 1136, dark ? C.forestRule : C.rule);
  text(page, "GROUP 4   /   PROJECT 15", 72, 660, 360, 25, { font: mono, size: 15, color });
  text(page, String(n).padStart(2, "0"), 1148, 660, 60, 25, { font: mono, size: 15, color, align: "right" });
  return page;
}

function title(page, label, dark = false) {
  text(page, label, 72, 55, 1136, 75, { font: serif, size: 53, color: dark ? C.sheet : C.ink });
}

function label(page, value, x, y, width, dark = false) {
  text(page, value, x, y, width, 26, { font: mono, size: 17, color: dark ? C.lime : C.warm });
}

function paragraph(page, value, x, y, width, height, dark = false, size = 25) {
  text(page, value, x, y, width, height, { size, color: dark ? C.soft : C.muted, wrap: true });
}

function grid(page, values, x, y, widths, rowHeight, options = {}) {
  const nativeTable = page.tables.add({
    rows: values.length, columns: widths.length,
    left: x, top: y, width: widths.reduce((a, b) => a + b, 0),
    height: values.length * rowHeight, columnWidths: widths, values,
  });
  nativeTable.styleOptions = { headerRow: false, bandedRows: false, firstColumn: false };
  nativeTable.borders.assign({ style: "solid", fill: C.rule, width: 1 });
  values.forEach((row, r) => {
    nativeTable.rows[r].height = rowHeight;
    row.forEach((_, c) => {
      const cell = nativeTable.getCell(r, c);
      const header = r === 0;
      const highlighted = r === options.highlightRow || (options.diagonal && r > 0 && r === c);
      cell.fill = header ? C.green : highlighted ? C.greenTint : C.sheet;
      cell.text.style = {
        typeface: body, fontSize: options.fontSize ?? 23, color: header ? C.sheet : C.ink,
        alignment: c === 0 ? "left" : options.align ?? "center", verticalAlignment: "middle",
        bold: header || highlighted, wrap: "none", autoFit: "none",
      };
    });
  });
  return nativeTable;
}

{
  const page = slide(true);
  label(page, "NLP CAPSTONE", 75, 80, 400, true);
  text(page, "Ledger", 66, 125, 1136, 200, { font: serif, size: 158, color: C.lime });
  text(page, "Domain-Specific Document Classifier", 76, 346, 1129, 64, { font: serif, size: 49, color: C.sheet });
  paragraph(page, "Routing consumer finance complaints by product", 78, 423, 960, 48, true, 27);
  page.speakerNotes.text = "Introduce Ledger, our Project 15 app. Ledger BERT is the fine-tuned transformer. The deployed app uses the Ledger ensemble.";
}

{
  const page = slide();
  title(page, "Group 4");
  grid(page, [["Student", "Roll number"], ...team], 72, 153, [633, 503], 63, { align: "left", fontSize: 26 });
  page.speakerNotes.text = "Group and roll numbers come from the assignment image. Names come from the student roster supplied with the assignment.";
}

{
  const page = slide();
  title(page, "Complaint routing");
  label(page, "THE TASK", 72, 163, 540);
  text(page, "Which team should\nread this complaint?", 68, 208, 600, 158, { font: serif, size: 61, wrap: true });
  paragraph(page, "A complaint arrives as a note or document. Staff need a first category before they can send it to a product queue.", 75, 404, 540, 130);
  label(page, "SUPPORTED PRODUCT QUEUES", 735, 163, 473);
  labels.forEach((entry, index) => {
    const y = 212 + index * 68;
    rule(page, 735, y, 473);
    text(page, entry, 735, y + 6, 473, 47, { font: serif, size: 32 });
  });
  page.speakerNotes.text = "A bank help desk or consumer complaint intake team can use the prediction as a routing suggestion. The original product labels are consumer-selected categories.\n" + dataSource;
}

{
  const page = slide();
  title(page, "Dataset and document length");
  label(page, "SOURCE AND SPLIT", 72, 155, 480);
  text(page, integer(data.total), 67, 186, 445, 97, { font: serif, size: 88, color: C.green });
  paragraph(page, "English CFPB narratives in the cleaned, balanced sample.", 72, 283, 458, 70);
  grid(page, [
    ["Train", "Validation", "Test"],
    [integer(data.splits.train), integer(data.splits.validation), integer(data.splits.test)],
  ], 72, 378, [145, 157, 137], 50, { fontSize: 21 });
  paragraph(page, "Sampled complaints span January 2019 to January 2020.", 72, 516, 455, 66, false, 20);
  page.charts.add("bar", {
    position: { left: 555, top: 157, width: 653, height: 408 },
    title: "Median words by product",
    titleTextStyle: { typeface: body, fontSize: 23, color: C.ink },
    categories: ["Bank account", "Credit card", "Credit reporting", "Debt collection", "Mortgage"],
    series: [{ name: "Median words", values: labels.map(name => data.by_class[name].median_words), fill: C.green }],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 62 },
    hasLegend: false,
    chartFill: C.paper, plotAreaFill: C.paper,
    dataLabels: { showValue: true, position: "outEnd", textStyle: { typeface: body, fontSize: 17 } },
    xAxis: { textStyle: { typeface: body, fontSize: 15 } },
    yAxis: { textStyle: { typeface: body, fontSize: 15 }, majorGridlines: { fill: C.rule, width: 1 } },
  });
  page.speakerNotes.text = `Archive size: ${integer(source.archive_rows)} public complaint records. Eligible category counts were heavily skewed toward credit reporting, so we took 450 cleaned narratives per category. Each test class has 68 complaints.\n${dataSource}`;
}

{
  const page = slide();
  title(page, "Cleaning and text features");
  const entries = [
    ["Prepare the corpus", "Normalize Unicode and spacing. Remove masking tokens, short text and repeated narratives before splitting."],
    ["TF-IDF baseline", "Extract word unigrams and bigrams. Fit IDF and multinomial Naive Bayes on the training split."],
    ["Transformer input", "Tokenize with uncased WordPiece. Keep 126 content tokens plus CLS and SEP."],
  ];
  entries.forEach(([heading, copy], index) => {
    const y = 166 + index * 137;
    rule(page, 72, y, 1136);
    text(page, heading, 72, y + 19, 430, 71, { font: serif, size: 35, color: C.green });
    paragraph(page, copy, 521, y + 14, 650, 88);
  });
  page.speakerNotes.text = `Removed ${data.cleaning.missing_or_short ?? 0} short or blank records and ${data.cleaning.same_label_duplicate ?? 0} same-label repeats. Removed all rows in exact-text groups that had conflicting labels. Keep wording and word order for BERT.\n${dataSource}\n${modelSource}`;
}

{
  const page = slide(true);
  title(page, "Fine-tuning Ledger BERT", true);
  const rows = [
    ["Pretrained encoder", "BERT-Tiny has two transformer blocks, two attention heads per block and a hidden width of 128."],
    ["Trainable layers", "Update the final feed-forward projection, its LayerNorm and a new five-class head."],
    ["Validation", `Select epoch ${training.selected_epoch} by macro F1. Tune score temperatures and the BERT / TF-IDF blend on validation data.`],
  ];
  rows.forEach(([heading, copy], index) => {
    const y = 166 + index * 141;
    rule(page, 72, y, 1136, C.forestRule);
    text(page, heading, 72, y + 20, 435, 71, { font: serif, size: 34, color: C.lime });
    paragraph(page, copy, 528, y + 16, 666, 89, true);
  });
  page.speakerNotes.text = `The classifier pools CLS and a mean of up to 31 content states. The frozen encoder prefix is computed once for CPU training. Cross-entropy gradients update the listed parameters with Adam. Final projection L2 change: ${training.encoder_projection_delta_l2.toFixed(3)}. The ensemble gives BERT ${Math.round(training.bert_weight * 100)}% of the weight after branch temperature scaling.\n${modelSource}`;
}

{
  const page = slide();
  title(page, "Held-out performance");
  paragraph(page, `${data.splits.test} complaints with 68 cases per product.`, 72, 139, 1080, 48);
  const keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"];
  grid(page, [
    ["Model", "Accuracy", "Precision", "Recall", "F1"],
    ...[["Ledger BERT", "bert"], ["TF-IDF + Naive Bayes", "baseline"], ["Ledger ensemble", "ensemble"]]
      .map(([name, key]) => [name, ...keys.map(metric => percent(report[key][metric]))]),
  ], 72, 219, [360, 188, 196, 196, 196], 70, { highlightRow: 3, fontSize: 23 });
  const diff = (report.ensemble.macro_f1 - report.baseline.macro_f1) * 100;
  const relation = diff >= 0 ? "ahead of" : "behind";
  paragraph(page, `The ensemble was ${Math.abs(diff).toFixed(1)} F1 points ${relation} TF-IDF on this split. Precision, recall and F1 are macro averages.`, 72, 521, 1110, 79, false, 23);
  page.speakerNotes.text = "All rows use the same test complaints. BERT reads a prefix and TF-IDF sees the complete narrative. Scores are from analysis/metrics.json and can be recomputed from test_predictions.csv.";
}

{
  const page = slide();
  title(page, "Where the model gets it wrong");
  const matrix = report.ensemble.confusion_matrix;
  grid(page, [
    ["Actual / predicted", "Bank", "Card", "Report", "Debt", "Mortgage"],
    ...labels.map((name, index) => [name, ...matrix[index].map(String)]),
  ], 72, 165, [272, 118, 118, 118, 118, 135], 57, { diagonal: true, fontSize: 21 });
  const top = matrix.flatMap((row, actual) => row.map((count, predicted) => ({ count, actual, predicted })))
    .filter(entry => entry.actual !== entry.predicted).sort((a, b) => b.count - a.count)[0];
  text(page, String(top.count), 1000, 181, 208, 100, { font: serif, size: 88, color: C.warm });
  paragraph(page, `${labels[top.actual]} complaints predicted as ${labels[top.predicted].toLowerCase()}.`, 997, 285, 211, 129, false, 22);
  paragraph(page, "Each row contains 68 complaints. Shaded cells show correct predictions.", 72, 551, 1030, 48, false, 22);
  page.speakerNotes.text = "Read the matrix by row. Product terms can overlap when a debtor disputes collections on a credit report, or when a bank product and its account reporting appear in the same narrative. The report also contains precision, recall and F1 for each class.";
}

{
  const page = slide();
  title(page, "Live demo");
  const image = await fs.readFile(path.join(root, "screenshots", "classification.png"));
  page.images.add({
    blob: image, contentType: "image/png",
    alt: "Ledger browser workspace with a credit card complaint and its ranked predictions",
    fit: "contain", position: { left: 72, top: 153, width: 815, height: 453 },
  });
  const steps = [
    ["Paste or load a sample", "Choose Card charge to fill the input."],
    ["Classify the complaint", "Read the category scores and open the summary."],
    ["Try a PDF upload", "Choose mortgage-escrow.pdf."],
  ];
  steps.forEach(([name, copy], index) => {
    const y = 166 + index * 142;
    rule(page, 917, y, 291);
    text(page, name, 917, y + 12, 291, 47, { font: serif, size: 24, color: C.green });
    paragraph(page, copy, 917, y + 62, 291, 57, false, 20);
  });
  page.speakerNotes.text = "Run python app.py and open http://127.0.0.1:8765. The screenshot comes from the running app with a fictional example. Demonstrate Card charge. Then switch to Upload file and open examples/mortgage-escrow.pdf. Show the review cue if it appears.";
}

{
  const page = slide(true);
  title(page, "Limits and next steps", true);
  text(page, percent(report.ensemble.macro_f1), 64, 172, 490, 155, { font: serif, size: 120, color: C.lime });
  paragraph(page, "Ensemble macro F1 on the balanced, held-out sample.", 75, 331, 480, 92, true);
  label(page, "LIMITS", 620, 179, 588, true);
  paragraph(page, "Five English complaint categories. The training archive predates current complaints, and long documents get partial BERT coverage.", 620, 224, 588, 104, true);
  rule(page, 620, 356, 588, C.forestRule);
  label(page, "NEXT EXPERIMENT", 620, 378, 588, true);
  paragraph(page, "Test on a later time period and measure how many out-of-scope complaints get flagged for human review.", 620, 418, 588, 124, true);
  page.speakerNotes.text = "Confidence is a temperature-scaled score and does not certify a correct route. Photo-only PDFs need OCR. Next, test on later intake data and explicitly measure errors on unsupported products.";
}

const preview = path.join(work, "slides");
await fs.mkdir(preview, { recursive: true });
await fs.mkdir(output, { recursive: true });
for (const [index, page] of pres.slides.items.entries()) {
  const png = await pres.export({ slide: page, format: "png", scale: 1 });
  await fs.writeFile(path.join(preview, `slide-${String(index + 1).padStart(2, "0")}.png`), new Uint8Array(await png.arrayBuffer()));
}
const draft = path.join(work, "ledger-candidate.pptx");
const finalPath = path.join(output, "Project 15 - Ledger.pptx");
const tableSlides = [2, 4, 7, 8];
const validationDir = await fs.mkdtemp(path.join(work, "slides-validation-"));
await (await PresentationFile.exportPptx(pres)).save(draft);
const result = await finalizePresentation({
  workspaceDir: root, candidatePath: draft, finalPath, pythonExecutable: python,
  integrityValidatorPath: path.join(skill, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skill, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", "12192000,6858000", "--validate-heading-fit",
    ...tableSlides.flatMap(number => ["--require-native-table-slide", String(number)]),
  ],
  explicitTotalSlideCount: 10,
  requiredNativeTableOwnerSlides: tableSlides,
  requiredNativeChartOwnerSlides: [4],
  fontPolicy: { basis: "design", families: [serif, body, mono] },
  verifyArtifactToolImport: true,
  materializeLiteralChartWorkbooks: true,
  receiptPath: path.join(validationDir, "receipt.json"),
});
console.log(result.finalPath ?? finalPath);
