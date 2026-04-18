const DEFAULT_TOP_K = 5;
const STORAGE_KEY = "gost-chat.chatSessions.v1";
const UNTITLED_SESSION_TITLE = "Новый чат";
const QUESTION_PREFIX_WORDS = new Set([
  "что",
  "как",
  "какие",
  "какой",
  "какая",
  "какое",
  "каких",
  "где",
  "когда",
  "почему",
  "зачем",
  "можно",
  "нужно",
]);

const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const messages = document.querySelector("#messages");
const errorBox = document.querySelector("#error");
const sendButton = document.querySelector("#send-button");
const newSessionButton = document.querySelector("#new-session-button");
const sessionList = document.querySelector("#session-list");

let requestInFlight = false;
let chatStore = loadChatStore();

function setError(message) {
  errorBox.textContent = message;
  errorBox.hidden = !message;
}

function createId(prefix) {
  if (window.crypto && window.crypto.randomUUID) {
    return `${prefix}-${window.crypto.randomUUID()}`;
  }

  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function createSession() {
  const now = new Date().toISOString();
  return {
    id: createId("session"),
    title: UNTITLED_SESSION_TITLE,
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
}

function createDefaultStore() {
  const session = createSession();
  return {
    activeSessionId: session.id,
    sessions: [session],
  };
}

function loadChatStore() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return createDefaultStore();
    }

    const parsed = JSON.parse(raw);
    if (!isValidStore(parsed)) {
      localStorage.removeItem(STORAGE_KEY);
      return createDefaultStore();
    }

    return normalizeStore(parsed);
  } catch (error) {
    localStorage.removeItem(STORAGE_KEY);
    return createDefaultStore();
  }
}

function isValidStore(value) {
  return (
    value &&
    typeof value === "object" &&
    typeof value.activeSessionId === "string" &&
    Array.isArray(value.sessions) &&
    value.sessions.every(isValidSession)
  );
}

function isValidSession(session) {
  return (
    session &&
    typeof session === "object" &&
    typeof session.id === "string" &&
    typeof session.title === "string" &&
    typeof session.createdAt === "string" &&
    typeof session.updatedAt === "string" &&
    Array.isArray(session.messages) &&
    session.messages.every(isValidMessage)
  );
}

function isValidMessage(message) {
  return (
    message &&
    typeof message === "object" &&
    typeof message.id === "string" &&
    (message.role === "user" || message.role === "assistant") &&
    typeof message.body === "string" &&
    typeof message.createdAt === "string"
  );
}

function normalizeStore(store) {
  const sessions = store.sessions.length ? store.sessions : [createSession()];
  const activeExists = sessions.some((session) => session.id === store.activeSessionId);
  const activeSessionId = activeExists ? store.activeSessionId : sessions[0].id;
  return { activeSessionId, sessions };
}

function saveChatStore() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chatStore));
    return true;
  } catch (error) {
    setError("Не удалось сохранить историю чатов в браузере.");
    return false;
  }
}

function getActiveSession() {
  return chatStore.sessions.find((session) => session.id === chatStore.activeSessionId);
}

function sortSessionsByUpdatedAt(sessions) {
  return [...sessions].sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
}

function renderApp() {
  renderSessionList();
  renderActiveSessionMessages();
  setLoading(requestInFlight);
}

function renderSessionList() {
  sessionList.replaceChildren();

  if (!chatStore.sessions.length) {
    const empty = document.createElement("p");
    empty.className = "session-empty";
    empty.textContent = "Чатов пока нет";
    sessionList.append(empty);
    return;
  }

  sortSessionsByUpdatedAt(chatStore.sessions).forEach((session) => {
    const item = document.createElement("div");
    item.className = "session-item";
    if (session.id === chatStore.activeSessionId) {
      item.classList.add("active");
    }

    const openButton = document.createElement("button");
    openButton.className = "session-open-button";
    openButton.type = "button";
    openButton.disabled = requestInFlight;
    openButton.addEventListener("click", () => switchSession(session.id));

    const title = document.createElement("span");
    title.className = "session-title";
    title.textContent = session.title || UNTITLED_SESSION_TITLE;

    const updatedAt = document.createElement("span");
    updatedAt.className = "session-updated";
    updatedAt.textContent = formatSessionTime(session.updatedAt);

    openButton.append(title, updatedAt);

    const deleteButton = document.createElement("button");
    deleteButton.className = "session-delete-button";
    deleteButton.type = "button";
    deleteButton.textContent = "Удалить чат";
    deleteButton.disabled = requestInFlight;
    deleteButton.addEventListener("click", () => deleteSession(session.id));

    item.append(openButton, deleteButton);
    sessionList.append(item);
  });
}

function renderActiveSessionMessages() {
  messages.replaceChildren();
  const activeSession = getActiveSession();
  if (!activeSession) {
    return;
  }

  activeSession.messages.forEach((message) => {
    if (message.role === "assistant" && message.askResult) {
      appendAssistantResult(message.askResult, message.elapsedSeconds);
      return;
    }

    appendMessage(message.role, message.body);
  });
}

function switchSession(sessionId) {
  if (requestInFlight || sessionId === chatStore.activeSessionId) {
    return;
  }

  chatStore.activeSessionId = sessionId;
  saveChatStore();
  setError("");
  renderApp();
  input.focus();
}

function createNewSession() {
  if (requestInFlight) {
    return;
  }

  const session = createSession();
  chatStore.sessions.push(session);
  chatStore.activeSessionId = session.id;
  saveChatStore();
  setError("");
  renderApp();
  input.focus();
}

function deleteSession(sessionId) {
  if (requestInFlight) {
    return;
  }

  chatStore.sessions = chatStore.sessions.filter((session) => session.id !== sessionId);

  if (!chatStore.sessions.length) {
    const session = createSession();
    chatStore.sessions = [session];
    chatStore.activeSessionId = session.id;
  } else if (chatStore.activeSessionId === sessionId) {
    chatStore.activeSessionId = sortSessionsByUpdatedAt(chatStore.sessions)[0].id;
  }

  saveChatStore();
  setError("");
  renderApp();
  input.focus();
}

function appendStoredMessage(session, message) {
  session.messages.push(message);
  session.updatedAt = message.createdAt;
}

function createStoredMessage(role, body, extra = {}) {
  return {
    id: createId("message"),
    role,
    body,
    createdAt: new Date().toISOString(),
    ...extra,
  };
}

function maybeRenameSessionFromFirstQuestion(session, question) {
  if (session.title !== UNTITLED_SESSION_TITLE) {
    return;
  }

  const hasPreviousUserMessage = session.messages.some((message) => message.role === "user");
  if (hasPreviousUserMessage) {
    return;
  }

  const title = createSessionTitle(question);
  if (title) {
    session.title = title;
  }
}

function createSessionTitle(question) {
  const normalized = question
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^[\s.,!?;:()[\]{}"']+|[\s.,!?;:()[\]{}"']+$/g, "");

  if (!normalized) {
    return "";
  }

  let words = normalized.split(" ");
  while (words.length > 1 && QUESTION_PREFIX_WORDS.has(cleanWord(words[0]))) {
    words = words.slice(1);
  }

  let title = words.slice(0, 5).join(" ");
  if (title.length > 60) {
    const shortened = title.slice(0, 60);
    const lastSpaceIndex = shortened.lastIndexOf(" ");
    title = lastSpaceIndex > 0 ? shortened.slice(0, lastSpaceIndex) : shortened;
    title = `${title}...`;
  }

  return title || UNTITLED_SESSION_TITLE;
}

function cleanWord(word) {
  return word.toLowerCase().replace(/^[\s.,!?;:()[\]{}"']+|[\s.,!?;:()[\]{}"']+$/g, "");
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

function appendAssistantResult(data, elapsedSeconds) {
  const article = appendMessage("assistant", data.answer || "");
  replaceAssistantMessage(article, data, elapsedSeconds);
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

  if (Array.isArray(data.visual_evidence) && data.visual_evidence.length) {
    bodyNode.append(createVisualEvidenceList(data.visual_evidence));
  }

  if (Array.isArray(data.citations) && data.citations.length) {
    bodyNode.append(createCitationList(data.citations));
  }

  scrollMessagesToBottom();
}

function createCitationSummary(data) {
  const summary = document.createElement("p");
  summary.className = "citation-summary";
  const citations = Array.isArray(data.citations) ? data.citations : [];

  if (!citations.length) {
    summary.textContent = "Источники не найдены.";
    return summary;
  }

  const count = citations.length;
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
    if (citation.visual_evidence) {
      item.append(createVisualEvidenceList([citation.visual_evidence]));
    }
    list.append(item);
  });

  details.append(list);
  return details;
}

function createVisualEvidenceList(items) {
  const visuals = Array.isArray(items) ? items.filter((item) => item && item.crop_url) : [];
  const wrapper = document.createElement("div");
  wrapper.className = "visual-evidence-list";

  visuals.forEach((visual) => {
    const figure = document.createElement("figure");
    figure.className = "visual-evidence";

    const image = document.createElement("img");
    image.src = visual.crop_url;
    image.alt = formatVisualAltText(visual);
    image.loading = "lazy";

    const caption = document.createElement("figcaption");
    caption.textContent = formatVisualCaption(visual);

    figure.append(image, caption);
    wrapper.append(figure);
  });

  return wrapper;
}

function formatVisualAltText(visual) {
  const label = visual.label || visual.block_type || "visual evidence";
  return `${label} from ${visual.source_file}, page ${visual.page_number}`;
}

function formatVisualCaption(visual) {
  const label = visual.label ? `${visual.label} - ` : "";
  return `${label}${visual.source_file}, page ${visual.page_number}`;
}

function setLoading(isLoading) {
  sendButton.disabled = isLoading;
  input.disabled = isLoading;
  newSessionButton.disabled = isLoading;
  sendButton.textContent = isLoading ? "Ждём..." : "Отправить";
  renderSessionList();
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

function formatSessionTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "без даты";
  }

  return date.toLocaleString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
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

newSessionButton.addEventListener("click", createNewSession);

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

  const activeSession = getActiveSession();
  if (!activeSession) {
    chatStore = createDefaultStore();
  }

  const session = getActiveSession();
  maybeRenameSessionFromFirstQuestion(session, message);

  const userMessage = createStoredMessage("user", message);
  appendStoredMessage(session, userMessage);
  saveChatStore();
  renderSessionList();

  appendMessage("user", message);
  input.value = "";

  const loadingMessage = appendMessage("assistant", "Ищу фрагменты в документах...");
  requestInFlight = true;
  setLoading(true);

  try {
    const result = await askIndexedDocuments(message);
    replaceAssistantMessage(loadingMessage, result, result.elapsedSeconds);

    const assistantMessage = createStoredMessage("assistant", result.answer, {
      askResult: result,
      elapsedSeconds: result.elapsedSeconds,
    });
    appendStoredMessage(session, assistantMessage);
    saveChatStore();
    renderSessionList();
  } catch (error) {
    loadingMessage.remove();
    setError(error.message);
  } finally {
    requestInFlight = false;
    setLoading(false);
    input.focus();
  }
});

saveChatStore();
renderApp();
