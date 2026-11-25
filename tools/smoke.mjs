import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.LEDGER_NODE_MODULES
  ? path.join(process.env.LEDGER_NODE_MODULES, "playwright")
  : "playwright");
const output = path.join(root, "screenshots");
const python = process.env.LEDGER_PYTHON ?? "python";
const child = spawn(python, ["app.py", "--port", "0"], {
  cwd: root, stdio: ["ignore", "pipe", "inherit"], windowsHide: true,
});
let browser;

function serverAddress() {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("Ledger did not start within 20 seconds")), 20_000);
    child.once("error", (error) => { clearTimeout(timer); reject(error); });
    child.once("exit", (code) => { clearTimeout(timer); reject(new Error(`Ledger exited with ${code}`)); });
    let startup = "";
    child.stdout.on("data", (chunk) => {
      startup += chunk.toString();
      const match = startup.match(/http:\/\/127\.0\.0\.1:\d+/);
      if (match) { clearTimeout(timer); resolve(match[0]); }
    });
  });
}

function pdfFixture(pages, password = "") {
  const result = spawnSync(python, ["-c", [
    "import sys",
    "from pypdf import PdfWriter",
    "writer = PdfWriter()",
    "for _ in range(int(sys.argv[1])): writer.add_blank_page(width=72, height=72)",
    "if sys.argv[2]: writer.encrypt(sys.argv[2])",
    "writer.write(sys.stdout.buffer)",
  ].join("\n"), String(pages), password], { cwd: root, windowsHide: true });
  assert.equal(result.status, 0, result.stderr?.toString());
  return result.stdout;
}

async function expectError(response, status, pattern) {
  assert.equal(response.status(), status);
  assert.match((await response.json()).error, pattern);
}

try {
  const address = await serverAddress();
  browser = await chromium.launch({ executablePath: process.env.LEDGER_CHROME || undefined });
  await mkdir(output, { recursive: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 800 } });
  const browserErrors = [];
  page.on("pageerror", error => browserErrors.push(error.message));
  await page.goto(address);
  await page.screenshot({ path: path.join(output, "workspace.png"), fullPage: true });
  const submit = page.getByRole("button", { name: "Classify document" });
  const complaint = page.getByRole("textbox", { name: "Complaint text" });
  const category = page.locator("#category");

  await submit.click();
  assert.match(await page.locator("#error").innerText(), /at least 8/);
  await complaint.fill("Word");
  assert.equal(await page.locator("#word-count").innerText(), "1 word");
  await page.getByRole("button", { name: "Card charge" }).click();
  await page.keyboard.press("Control+Enter");
  await category.waitFor();
  assert.equal(await category.innerText(), "Credit card");
  assert.equal(await page.locator("#scores li").count(), 5);
  const summary = page.getByText("Document summary", { exact: true });
  await summary.click();
  const excerpts = await page.locator("#summary-excerpts p").allTextContents();
  assert.ok(excerpts.length >= 1);
  for (const excerpt of excerpts) assert.ok((await complaint.inputValue()).includes(excerpt));
  await summary.click();
  await page.mouse.move(7, 7);
  await page.screenshot({ path: path.join(output, "classification.png"), fullPage: true });

  await page.getByRole("button", { name: "Model notes" }).click();
  await page.getByRole("dialog").waitFor();
  await page.waitForFunction(() => document.querySelector("#model-size").textContent === "340");
  assert.equal(await page.locator("#model-f1").innerText(), "83.6%");
  await page.keyboard.press("Escape");
  await page.getByRole("dialog").waitFor({ state: "hidden" });
  await page.getByRole("button", { name: "Model notes" }).click();
  await page.getByRole("button", { name: "Close model notes" }).click();
  await page.getByRole("dialog").waitFor({ state: "hidden" });

  const extraCases = [
    ["My checking account was charged several overdraft fees after a pending debit card purchase posted. My bank would not reverse the fees or explain the account balance.", "Bank account"],
    ["A collection agency keeps calling me about a debt I do not recognize. I asked them to send written validation of the debt and stop calling my workplace, but the calls have continued.", "Debt collection"],
  ];
  for (const [text, expected] of extraCases) {
    await complaint.fill(text);
    await submit.click();
    await category.waitFor();
    assert.equal(await category.innerText(), expected);
  }
  await complaint.fill("I contacted the company last week and asked for written confirmation, but no one has explained what happened or responded to my message.");
  await submit.click();
  await page.locator("#review").waitFor();
  assert.match(await page.locator("#review").innerText(), /Review suggested/);

  const classifierRoute = "**/api/classify";
  await page.route(classifierRoute, route => route.abort("failed"));
  await submit.click();
  await page.locator("#error").waitFor();
  assert.match(await page.locator("#error").innerText(), /Could not reach/);
  await page.unroute(classifierRoute);
  await submit.click();
  await category.waitFor();

  let releaseResponse;
  const holdResponse = new Promise(resolve => { releaseResponse = resolve; });
  await page.route(classifierRoute, async route => {
    await holdResponse;
    await route.abort().catch(() => {});
  });
  await submit.click();
  await page.getByText("Classifying…", { exact: true }).waitFor();
  assert.ok(await submit.isDisabled());
  await page.getByRole("button", { name: "Clear", exact: true }).click();
  releaseResponse();
  await page.unrouteAll({ behavior: "wait" });
  assert.equal(await complaint.inputValue(), "");
  assert.equal(await category.isVisible(), false);
  assert.ok(await submit.isEnabled());
  assert.ok(await page.locator("#error").isHidden());

  const pasteTab = page.getByRole("tab", { name: "Paste text" });
  const uploadTab = page.getByRole("tab", { name: "Upload file" });
  await pasteTab.focus();
  await page.keyboard.press("End");
  assert.equal(await uploadTab.getAttribute("aria-selected"), "true");
  await page.keyboard.press("Home");
  assert.equal(await pasteTab.getAttribute("aria-selected"), "true");
  await page.keyboard.press("ArrowRight");
  await submit.click();
  assert.match(await page.locator("#error").innerText(), /Choose a document/);
  const chooser = page.waitForEvent("filechooser");
  await page.getByRole("button", { name: "Choose file", exact: true }).click();
  await (await chooser).setFiles(path.join(root, "examples", "mortgage-escrow.txt"));
  assert.equal(await page.locator("#file-name").innerText(), "mortgage-escrow.txt");
  const input = page.locator("#file");
  const cardText = await readFile(path.join(root, "examples", "card-charge.txt"), "utf8");
  const dragData = await page.evaluateHandle(content => {
    const transfer = new DataTransfer();
    transfer.items.add(new File([content], "dropped.txt", { type: "text/plain" }));
    return transfer;
  }, cardText);
  const dropzone = page.locator("#dropzone");
  await dropzone.dispatchEvent("dragover", { dataTransfer: dragData });
  assert.equal(await dropzone.getAttribute("data-dragging"), "true");
  await dropzone.dispatchEvent("dragleave", { dataTransfer: dragData });
  assert.equal(await dropzone.getAttribute("data-dragging"), null);
  await dropzone.dispatchEvent("drop", { dataTransfer: dragData });
  await dragData.dispose();
  await submit.click();
  await category.waitFor();
  assert.equal(await category.innerText(), "Credit card");
  await input.setInputFiles(path.join(root, "examples", "mortgage-escrow.txt"));
  assert.equal(await page.locator("#file-name").innerText(), "mortgage-escrow.txt");

  for (const [filename, expected] of [
    ["mortgage-escrow.txt", "Mortgage"],
    ["mortgage-escrow.pdf", "Mortgage"],
    ["card-charge.docx", "Credit card"],
  ]) {
    await input.setInputFiles(path.join(root, "examples", filename));
    await submit.click();
    await category.waitFor();
    assert.equal(await category.innerText(), expected, filename);
  }
  await page.mouse.move(7, 7);
  await page.screenshot({ path: path.join(output, "upload.png"), fullPage: true });

  for (const [name, buffer, pattern, submitFile] of [
    ["wrong.csv", Buffer.from("row"), /Choose a TXT/, false],
    ["empty.txt", Buffer.alloc(0), /non-empty/, false],
    ["oversize.txt", Buffer.alloc(8 * 1024 * 1024 + 1), /no larger than 8 MB/, false],
    ["broken.pdf", Buffer.from("This is not a PDF."), /PDF could not be read/, true],
    ["broken.docx", Buffer.from("This is not a DOCX."), /DOCX could not be read/, true],
    ["no-text.pdf", pdfFixture(1), /No readable text/, true],
    ["locked.pdf", pdfFixture(1, "sample"), /Unlock the PDF/, true],
    ["long.pdf", pdfFixture(41), /at most 40 pages/, true],
  ]) {
    await input.setInputFiles({ name, mimeType: "application/octet-stream", buffer });
    if (submitFile) await submit.click();
    await page.locator("#error").waitFor();
    assert.match(await page.locator("#error").innerText(), pattern);
  }
  await page.getByRole("button", { name: "Mortgage escrow" }).click();
  assert.equal(await pasteTab.getAttribute("aria-selected"), "true");
  await submit.click();
  await category.waitFor();
  assert.equal(await category.innerText(), "Mortgage");

  const mobile = await browser.newPage({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true, deviceScaleFactor: 1 });
  mobile.on("pageerror", error => browserErrors.push(error.message));
  await mobile.goto(address);
  await mobile.getByRole("button", { name: "Credit report" }).click();
  await mobile.getByRole("button", { name: "Classify document" }).click();
  await mobile.locator("#category").waitFor();
  assert.equal(await mobile.locator("#category").innerText(), "Credit reporting");
  await mobile.screenshot({ path: path.join(output, "mobile.png"), fullPage: true });
  for (const width of [320, 390, 768, 1024, 1440]) {
    const surface = width < 768 ? mobile : page;
    await surface.setViewportSize({ width, height: 844 });
    assert.ok(await surface.evaluate(() => document.documentElement.scrollWidth === innerWidth), `Horizontal overflow at ${width}px`);
  }

  const endpoint = `${address}/api/classify`;
  const results = await page.request.post(endpoint, { data: { text: cardText.repeat(20) } });
  assert.equal(results.status(), 200);
  const long = await results.json();
  assert.equal(long.truncated, true);
  assert.equal(long.analyzed_tokens, 126);
  assert.ok(Math.abs(long.scores.reduce((sum, { score }) => sum + score, 0) - 1) < 1e-6);
  assert.ok(long.scores.every((entry, index) => !index || long.scores[index - 1].score >= entry.score));
  const textUpload = (name, data) => ({
    headers: { "Content-Type": "application/octet-stream", "X-Filename": name }, data,
  });
  const utf16 = Buffer.concat([Buffer.from([0xff, 0xfe]), Buffer.from(cardText, "utf16le")]);
  const upload = await page.request.post(endpoint, textUpload("example.TXT", utf16));
  assert.equal(upload.status(), 200);
  assert.equal((await upload.json()).category, "Credit card");
  await expectError(await page.request.post(endpoint, textUpload("invalid.txt", Buffer.from([0xff]))), 422, /Save the TXT/);
  await expectError(await page.request.post(endpoint, textUpload("invalid.csv", Buffer.from("row"))), 422, /Choose a TXT/);
  await expectError(await page.request.post(endpoint, {
    headers: { "Content-Type": "application/json" }, data: Buffer.from("{"),
  }), 400, /could not be read/);
  await expectError(await page.request.post(endpoint, { data: { text: 42 } }), 422, /as text/);
  await expectError(await page.request.post(endpoint, { data: { text: "too short" } }), 422, /at least 8/);
  await expectError(await page.request.post(endpoint, { data: { text: "word ".repeat(8001) } }), 422, /40,000/);
  await expectError(await page.request.post(endpoint, { headers: { "Content-Type": "text/plain" }, data: "plain" }), 415, /supported document/);
  await expectError(await page.request.get(`${address}/unknown`), 404, /Page not found/);
  assert.deepEqual(browserErrors, []);
  console.log("Passed: all five categories, samples, keyboard, summary, review, reset, retries, model notes, drag and drop, TXT/PDF/DOCX, validation, responsive widths, and API responses.");
} finally {
  await browser?.close();
  child.kill();
}
