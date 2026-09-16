const textArea = document.getElementById("input-text");
const encoderSelect = document.getElementById("encoder-select");
const maxLengthInput = document.getElementById("max-length");
const strideInput = document.getElementById("stride");
const lockStride = document.getElementById("lock-stride");
const viewSelect = document.getElementById("view-select");
const sampleBtn = document.getElementById("sample-btn");
const statusEl = document.getElementById("status");
const overlapNote = document.getElementById("overlap-note");
const rowsEl = document.getElementById("rows");

async function loadExample(exampleId) {
  const response = await fetch(`api/examples/${exampleId}`);
  const data = await response.json();
  // Examples are whole chapters; the first slice of one is plenty to see
  // the windowing, and keeps the page from rendering thousands of rows.
  textArea.value = data.text.trim().slice(0, 600);
}

// Stride only does anything to windows -- pairs advance one token at a
// time by definition -- so grey the control out rather than leaving a
// live-looking knob that changes nothing.
function syncStride() {
  const strideApplies = viewSelect.value !== "pairs";
  if (lockStride.checked) {
    strideInput.value = maxLengthInput.value;
  }
  strideInput.disabled = lockStride.checked || !strideApplies;
  lockStride.disabled = !strideApplies;
  document.querySelector(".stride-label").classList.toggle(
    "inactive",
    !strideApplies,
  );
}

async function sample() {
  const maxLength = Number(maxLengthInput.value);
  const stride = Number(strideInput.value);
  statusEl.textContent = "sampling...";
  const response = await fetch("api/sample", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: textArea.value,
      encoder: encoderSelect.value,
      max_length: maxLength,
      stride: stride,
      view: viewSelect.value,
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    statusEl.textContent = data.error || "something went wrong";
    rowsEl.innerHTML = "";
    overlapNote.textContent = "";
    return;
  }
  render(data);
}

function render(data) {
  const noun = data.view === "pairs" ? "pair" : "window";
  const shown = data.rows.length;
  statusEl.textContent =
    `${data.token_count} token${data.token_count === 1 ? "" : "s"} → ` +
    `${shown}${data.truncated ? "+" : ""} ${noun}${shown === 1 ? "" : "s"}` +
    (data.truncated ? ` (showing the first ${shown})` : "");

  overlapNote.textContent = data.view === "pairs" ? "" : describeStride(data);

  rowsEl.innerHTML = "";
  for (const row of data.rows) {
    rowsEl.appendChild(
      data.view === "pairs" ? renderPair(row) : renderWindow(row),
    );
  }
  if (!data.rows.length) {
    const empty = document.createElement("p");
    empty.className = "note";
    empty.textContent =
      data.view === "pairs"
        ? "Not enough tokens to make a pair — two is the minimum."
        : `Not enough tokens for a window — ${data.max_length} + 1 is the minimum.`;
    rowsEl.appendChild(empty);
  }
}

// Stride only means something relative to the context size, so say what
// the current combination actually does to the text rather than making
// the reader work it out from the rows.
function describeStride({ max_length: maxLength, stride }) {
  if (stride === maxLength) {
    return `Stride = context size: windows tile the text edge to edge. Every token is predicted exactly once.`;
  }
  if (stride < maxLength) {
    const overlap = maxLength - stride;
    return `Stride < context size: each window repeats ${overlap} token${overlap === 1 ? "" : "s"} of the previous one. More windows, but the overlap re-trains predictions the last window already covered.`;
  }
  const skipped = stride - maxLength;
  return `Stride > context size: ${skipped} token${skipped === 1 ? "" : "s"} between windows are skipped entirely, never used for training.`;
}

function tokenChips(tokens, className) {
  const wrap = document.createElement("div");
  wrap.className = "chips";
  for (const { id, text } of tokens) {
    const chip = document.createElement("span");
    chip.className = className;

    const idSpan = document.createElement("span");
    idSpan.className = "chip-id";
    idSpan.textContent = id;

    const textSpan = document.createElement("span");
    textSpan.className = "chip-text";
    // Spaces are part of a BPE token; make them visible rather than
    // letting the browser collapse them into nothing.
    textSpan.textContent = text.replace(/ /g, "·").replace(/\n/g, "⏎");

    chip.append(idSpan, textSpan);
    wrap.appendChild(chip);
  }
  return wrap;
}

function labelled(label, node) {
  const line = document.createElement("div");
  line.className = "line";
  const labelEl = document.createElement("span");
  labelEl.className = "line-label";
  labelEl.textContent = label;
  line.append(labelEl, node);
  return line;
}

function textNode(text) {
  const el = document.createElement("span");
  el.className = "as-text";
  el.textContent = text;
  return el;
}

function renderWindow(row) {
  const card = document.createElement("div");
  card.className = "row";

  const head = document.createElement("div");
  head.className = "row-head";
  head.textContent = `tokens ${row.start}–${row.start + row.inputs.length - 1}`;
  card.appendChild(head);

  card.appendChild(labelled("x", tokenChips(row.inputs, "chip chip-input")));
  card.appendChild(labelled("y", tokenChips(row.targets, "chip chip-target")));
  card.appendChild(labelled("", textNode(row.inputs_text)));
  card.appendChild(labelled("→", textNode(row.targets_text)));
  return card;
}

function renderPair(row) {
  const card = document.createElement("div");
  card.className = "row";
  card.appendChild(
    labelled("context", tokenChips(row.context, "chip chip-input")),
  );
  card.appendChild(
    labelled("target", tokenChips([row.target], "chip chip-target")),
  );
  card.appendChild(
    labelled("", textNode(`${row.context_text} → ${row.target_text}`)),
  );
  return card;
}

sampleBtn.addEventListener("click", sample);
maxLengthInput.addEventListener("input", syncStride);
lockStride.addEventListener("change", syncStride);
viewSelect.addEventListener("change", syncStride);
document.querySelectorAll(".example-btn").forEach((btn) => {
  btn.addEventListener("click", () => loadExample(btn.dataset.exampleId));
});

syncStride();
