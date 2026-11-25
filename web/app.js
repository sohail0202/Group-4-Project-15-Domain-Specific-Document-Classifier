const samples = {
  card: "My credit card statement includes a charge from a merchant I have never used. I disputed the transaction with the card issuer, but the charge appeared again on my next billing statement and I was charged interest on it.",
  credit: "My credit report lists an account that I never opened. I submitted a dispute to the credit bureau and provided identification, but the account still appears on the report as delinquent. I want the record corrected.",
  mortgage: "My mortgage servicer increased my monthly payment after recalculating escrow. The escrow statement lists my property tax twice, and the servicer has not corrected the balance after I sent the tax receipts.",
};

const get = (id) => document.getElementById(id);
const form = get("classify-form");
const complaint = get("complaint");
const fileInput = get("file");
const dropzone = get("dropzone");
const submit = get("submit");
const errorMessage = get("error");
const status = get("result-status");
const modelDialog = get("model-dialog");
const tabs = [get("text-mode"), get("file-mode")];
const panels = [get("text-panel"), get("file-panel")];
let mode = 0;
let file = null;
let pending = null;

function setStatus(label, state = "ready") {
  get("status-text").textContent = label;
  status.dataset.state = state;
}

function showError(message = "") {
  errorMessage.hidden = !message;
  errorMessage.textContent = message;
}

function resetOutput() {
  pending?.abort();
  pending = null;
  submit.disabled = false;
  get("prediction").hidden = true;
  get("empty").hidden = false;
  setStatus("Ready");
  showError();
}

function setMode(index) {
  if (mode !== index) resetOutput();
  mode = index;
  tabs.forEach((tab, position) => {
    const active = position === mode;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    panels[position].hidden = !active;
  });
}

function countWords(text) {
  return (text.match(/[a-z]+(?:'[a-z]+)?/gi) ?? []).length;
}

function updateInput() {
  const count = countWords(complaint.value);
  get("word-count").textContent = `${count.toLocaleString()} ${count === 1 ? "word" : "words"}`;
  get("clear").disabled = !complaint.value && !file;
}

function selectFile(candidate) {
  resetOutput();
  file = null;
  if (candidate && !/\.(pdf|txt|docx)$/i.test(candidate.name)) {
    showError("Choose a TXT, PDF or DOCX file.");
  } else if (candidate && (!candidate.size || candidate.size > 8 * 1024 * 1024)) {
    showError("Choose a non-empty file no larger than 8 MB.");
  } else {
    file = candidate;
  }
  // The JS file survives clearing the picker, so choosing the same path always fires change.
  fileInput.value = "";
  get("file-name").textContent = file?.name ?? "Drop a document here";
  get("file-hint").textContent = file
    ? `${Math.max(1, Math.round(file.size / 1024)).toLocaleString()} KB`
    : "TXT, text-based PDF or DOCX · up to 8 MB";
  get("choose-file").textContent = file ? "Choose another file" : "Choose file";
  updateInput();
}

function percent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function displayResult(data) {
  get("category").textContent = data.category;
  get("confidence").textContent = percent(data.confidence);
  get("confidence-ring").style.setProperty("--confidence", percent(data.confidence));
  get("review").hidden = !data.needs_review;
  const scores = data.scores.map(({ label, score }) => {
    const item = document.createElement("li");
    item.className = "score-row";
    const name = document.createElement("span");
    name.textContent = label;
    const bar = document.createElement("span");
    bar.className = "bar";
    bar.setAttribute("aria-hidden", "true");
    const fill = document.createElement("span");
    fill.style.width = percent(score);
    bar.append(fill);
    const value = document.createElement("span");
    value.className = "score-value";
    value.textContent = percent(score);
    item.append(name, bar, value);
    return item;
  });
  get("scores").replaceChildren(...scores);
  get("token-note").textContent = data.truncated
    ? `BERT read the first ${data.analyzed_tokens} tokens. TF-IDF read the full text.`
    : `${data.word_count.toLocaleString()} words analyzed`;
  get("summary").open = false;
  get("summary-excerpts").replaceChildren(...data.summary.map((excerpt) => {
    const paragraph = document.createElement("p");
    paragraph.textContent = excerpt;
    return paragraph;
  }));
  get("empty").hidden = true;
  get("prediction").hidden = false;
  setStatus("Complete", "complete");
}

tabs.forEach((tab, index) => {
  tab.addEventListener("click", () => setMode(index));
  tab.addEventListener("keydown", (event) => {
    const destinations = { ArrowLeft: 1 - index, ArrowRight: 1 - index, Home: 0, End: 1 };
    if (!(event.key in destinations)) return;
    event.preventDefault();
    const destination = destinations[event.key];
    setMode(destination);
    tabs[destination].focus();
  });
});

complaint.addEventListener("input", () => {
  resetOutput();
  updateInput();
});
get("choose-file").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => selectFile(fileInput.files[0] ?? null));
dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.dataset.dragging = "true";
});
dropzone.addEventListener("dragleave", (event) => {
  if (!dropzone.contains(event.relatedTarget)) delete dropzone.dataset.dragging;
});
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  delete dropzone.dataset.dragging;
  selectFile(event.dataTransfer.files[0] ?? null);
});

get("clear").addEventListener("click", () => {
  complaint.value = "";
  selectFile(null);
  (mode === 0 ? complaint : get("choose-file")).focus();
});

document.querySelectorAll("[data-sample]").forEach((button) => {
  button.addEventListener("click", () => {
    setMode(0);
    resetOutput();
    complaint.value = samples[button.dataset.sample];
    updateInput();
    complaint.focus();
    complaint.setSelectionRange(0, 0);
    complaint.scrollTop = 0;
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (pending) return;
  resetOutput();
  if (mode === 0 && countWords(complaint.value) < 8) {
    showError("Enter a complaint with at least 8 English words.");
    complaint.focus();
    return;
  }
  if (mode === 1 && !file) {
    showError("Choose a document to classify.");
    get("choose-file").focus();
    return;
  }
  const controller = new AbortController();
  pending = controller;
  submit.disabled = true;
  setStatus("Classifying…", "running");
  const headers = mode === 0
    ? { "Content-Type": "application/json" }
    : { "Content-Type": "application/octet-stream", "X-Filename": encodeURIComponent(file.name) };
  const body = mode === 0 ? JSON.stringify({ text: complaint.value }) : file;
  try {
    const response = await fetch("/api/classify", {
      method: "POST", headers, body, signal: controller.signal,
    });
    const result = await response.json();
    if (pending !== controller) return;
    if (response.ok) {
      displayResult(result);
    } else {
      setStatus("Ready");
      showError(result.error ?? "The document could not be classified.");
    }
  } catch (error) {
    if (pending === controller && error.name !== "AbortError") {
      setStatus("Ready");
      showError("Could not reach the local classifier. Check that Ledger is running and try again.");
    }
  } finally {
    if (pending === controller) {
      pending = null;
      submit.disabled = false;
    }
  }
});

form.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    form.requestSubmit();
  }
});

get("open-model").addEventListener("click", async () => {
  modelDialog.showModal();
  try {
    const response = await fetch("/api/model");
    if (!response.ok) throw new Error("Model info unavailable");
    const data = await response.json();
    get("model-f1").textContent = percent(data.macro_f1);
    get("model-size").textContent = data.test_count.toLocaleString();
  } catch {
    get("model-f1").textContent = "Unavailable";
    get("model-size").textContent = "Unavailable";
  }
});
