const DEFAULT_TOP_K = 5;

const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const messages = document.querySelector("#messages");
const errorBox = document.querySelector("#error");
const sendButton = document.querySelector("#send-button");
let requestInFlight = false;

function setError(message) {
  errorBox.textContent = message;
  errorBox.hidden = !message;
}

function appendMessage(role, body) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const roleNode = document.createElement("div");
  roleNode.className = "message-role";
  roleNode.textContent = role === "user" ? "Вы" : "Ассистент";

  const bodyNode = document.createElement("div");
  bodyNode.className = "message-body";
  bodyNode.textContent = body;

  article.append(roleNode, bodyNode);
  messages.append(article);
  scrollMessagesToBottom();

  return article;
}

function replaceAssistantMessage(article, data, elapsedSeconds) {
  const bodyNode = article.querySelector(".message-body");
  bodyNode.replaceChildren();

  const answer = document.createElement("p");
  answer.className = "answer-text";
  answer.textContent = data.answer;
  bodyNode.append(answer);

  if (typeof elapsedSeconds === "number") {
    const duration = document.createElement("p");
    duration.className = "response-duration";
    duration.textContent = formatElapsedSeconds(elapsedSeconds);
    bodyNode.append(duration);
  }

  bodyNode.append(createCitationSummary(data));

  if (data.citations.length) {
    bodyNode.append(createCitationList(data.citations));
  }

  scrollMessagesToBottom();
}

function createCitationSummary(data) {
  const summary = document.createElement("p");
  summary.className = "citation-summary";

  if (!data.citations.length) {
    summary.textContent = "Источники не найдены.";
    return summary;
  }

  const count = data.citations.length;
  summary.textContent = `Источников: ${count}. Найдено фрагментов: ${data.retrieved_results_count}.`;
  return summary;
}

function createCitationList(citations) {
  const details = document.createElement("details");
  details.className = "citations";
  details.open = true;

  const summary = document.createElement("summary");
  summary.textContent = "Источники";
  details.append(summary);

  const list = document.createElement("ol");
  list.className = "citation-list";

  citations.forEach((citation) => {
    const item = document.createElement("li");

    const meta = document.createElement("div");
    meta.className = "citation-meta";
    meta.textContent = `${citation.file_name} - ${formatPageRange(citation)} - ${citation.chunk_id} - оценка ${formatScore(citation.score)}`;

    const preview = document.createElement("p");
    preview.className = "citation-preview";
    preview.textContent = citation.evidence_preview;

    item.append(meta, preview);
    list.append(item);
  });

  details.append(list);
  return details;
}

function setLoading(isLoading) {
  sendButton.disabled = isLoading;
  input.disabled = isLoading;
  sendButton.textContent = isLoading ? "Ждём..." : "Отправить";
}

function formatPageRange(result) {
  if (result.page_start === null || result.page_start === undefined) {
    return "страница неизвестна";
  }

  if (result.page_end && result.page_end !== result.page_start) {
    return `страницы ${result.page_start}-${result.page_end}`;
  }

  return `страница ${result.page_start}`;
}

function formatScore(score) {
  if (typeof score !== "number") {
    return "неизвестно";
  }

  return score.toFixed(4);
}

function formatElapsedSeconds(seconds) {
  if (seconds < 10) {
    return `${seconds.toFixed(2)} s`;
  }

  return `${seconds.toFixed(1)} s`;
}

function scrollMessagesToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

async function askIndexedDocuments(query) {
  const startedAt = performance.now();
  const response = await fetch("/ask", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ query, top_k: DEFAULT_TOP_K }),
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Не удалось ответить по проиндексированным документам.");
  }

  return {
    ...data,
    elapsedSeconds: (performance.now() - startedAt) / 1000,
  };
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (requestInFlight) {
    return;
  }

  setError("");

  const message = input.value.trim();
  if (!message) {
    setError("Вопрос не может быть пустым.");
    return;
  }

  appendMessage("user", message);
  input.value = "";

  const loadingMessage = appendMessage("assistant", "Ищу фрагменты в документах...");
  requestInFlight = true;
  setLoading(true);

  try {
    const result = await askIndexedDocuments(message);
    replaceAssistantMessage(loadingMessage, result, result.elapsedSeconds);
  } catch (error) {
    loadingMessage.remove();
    setError(error.message);
  } finally {
    requestInFlight = false;
    setLoading(false);
    input.focus();
  }
});
