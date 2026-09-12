const textArea = document.getElementById("input-text");
const tokenizeBtn = document.getElementById("tokenize-btn");
const tokenCount = document.getElementById("token-count");
const tokensEl = document.getElementById("tokens");

// Tokens that start with a letter/digit are rendered as "word" chips;
// everything else (punctuation, "--", etc.) gets a distinct style.
const WORD_START = /^[A-Za-z0-9]/;

async function tokenize() {
  const response = await fetch("api/tokenize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: textArea.value }),
  });
  const data = await response.json();
  renderTokens(data.tokens);
  tokenCount.textContent = `${data.count} token${data.count === 1 ? "" : "s"}`;
}

function renderTokens(tokens) {
  tokensEl.innerHTML = "";
  for (const token of tokens) {
    const span = document.createElement("span");
    span.className = WORD_START.test(token) ? "token" : "token punct";
    span.textContent = token;
    tokensEl.appendChild(span);
  }
}

async function loadExample(exampleId) {
  const response = await fetch(`api/examples/${exampleId}`);
  const data = await response.json();
  textArea.value = data.text;
  tokenize();
}

tokenizeBtn.addEventListener("click", tokenize);

document.querySelectorAll(".example-btn").forEach((btn) => {
  btn.addEventListener("click", () => loadExample(btn.dataset.exampleId));
});
