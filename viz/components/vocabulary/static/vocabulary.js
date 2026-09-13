const textArea = document.getElementById("input-text");
const loadAllBtn = document.getElementById("load-all-btn");
const buildBtn = document.getElementById("build-btn");
const vocabSize = document.getElementById("vocab-size");
const filterInput = document.getElementById("filter-input");
const entriesEl = document.getElementById("vocab-entries");

const exampleIds = Array.from(document.querySelectorAll(".example-btn")).map(
  (btn) => btn.dataset.exampleId,
);

// The full vocabulary from the last build, kept around so the filter box
// can slice it locally instead of round-tripping to the server.
let currentVocabulary = [];

async function fetchExampleText(exampleId) {
  const response = await fetch(`api/examples/${exampleId}`);
  const data = await response.json();
  return data.text;
}

async function loadExample(exampleId) {
  const text = await fetchExampleText(exampleId);
  const separator = textArea.value.trim() ? "\n\n" : "";
  textArea.value += separator + text;
}

async function loadAllExamples() {
  const texts = await Promise.all(exampleIds.map(fetchExampleText));
  textArea.value = texts.join("\n\n");
}

async function buildVocabulary() {
  const response = await fetch("api/build", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: textArea.value }),
  });
  const data = await response.json();
  currentVocabulary = data.vocabulary;
  vocabSize.textContent = `${data.size} unique token${data.size === 1 ? "" : "s"}`;
  filterInput.disabled = false;
  filterInput.value = "";
  renderEntries(currentVocabulary);
}

function renderEntries(entries) {
  entriesEl.innerHTML = "";
  for (const { token, id } of entries) {
    const row = document.createElement("div");
    row.className = "vocab-entry";

    const idSpan = document.createElement("span");
    idSpan.className = "vocab-id";
    idSpan.textContent = id;

    const tokenSpan = document.createElement("span");
    tokenSpan.className = "vocab-token";
    tokenSpan.textContent = token;

    row.append(idSpan, tokenSpan);
    entriesEl.appendChild(row);
  }
}

function applyFilter() {
  const needle = filterInput.value.toLowerCase();
  const filtered = needle
    ? currentVocabulary.filter(({ token }) => token.toLowerCase().includes(needle))
    : currentVocabulary;
  renderEntries(filtered);
}

buildBtn.addEventListener("click", buildVocabulary);
loadAllBtn.addEventListener("click", loadAllExamples);
filterInput.addEventListener("input", applyFilter);

document.querySelectorAll(".example-btn").forEach((btn) => {
  btn.addEventListener("click", () => loadExample(btn.dataset.exampleId));
});
