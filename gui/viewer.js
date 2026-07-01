const DEFAULT_VIDEO = "../downloads/yeast_protocol_1min_gui.mp4";
const DEFAULT_ANNOTATIONS = "../downloads/yeast_protocol_1min.annotations.json";
const ACTIVE_WINDOW_SECONDS = 1.35;

const video = document.querySelector("#video");
const overlay = document.querySelector("#overlay");
const timeline = document.querySelector("#timeline");
const annotationList = document.querySelector("#annotationList");
const timecode = document.querySelector("#timecode");
const videoTitle = document.querySelector("#videoTitle");
const videoFile = document.querySelector("#videoFile");
const annotationFile = document.querySelector("#annotationFile");

let annotations = [];
let activeAnnotationKeys = new Set();

const toSeconds = (value) => Number.isFinite(Number(value)) ? Number(value) : 0;
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const annotationDuration = () =>
  Math.max(...annotations.map((item) => item.timestamp), 1);
const mediaDuration = () =>
  Number.isFinite(video.duration) && video.duration > 0
    ? video.duration
    : annotationDuration();

const formatTime = (seconds) => {
  const safe = Math.max(0, seconds || 0);
  const minutes = Math.floor(safe / 60);
  const secs = (safe % 60).toFixed(1).padStart(4, "0");
  return `${String(minutes).padStart(2, "0")}:${secs}`;
};

const severityClass = (severity = "") =>
  severity.toLowerCase().replace(/\s+/g, "-");

const normalize = (data) => [
  ...(data.reproducibility_risks || []).map((item, index) => ({
    id: `risk-${index}`,
    type: "risk",
    timestamp: toSeconds(item.timestamp_sec),
    title: `${item.severity || "Risk"}: ${item.action || "Reproducibility risk"}`,
    body: item.issue || "",
    detail: item.suggested_fix ? `Fix: ${item.suggested_fix}` : "",
    confidence: item.confidence,
    severity: item.severity || "Medium",
    className: `risk ${severityClass(item.severity || "Medium")}`,
  })),
  ...(data.thumbs_up || []).map((item, index) => ({
    id: `good-${index}`,
    type: "good",
    timestamp: toSeconds(item.timestamp_sec),
    title: `Thumbs up: ${item.practice || "Good practice"}`,
    body: item.why_it_helps || "",
    detail: "",
    confidence: item.confidence,
    severity: "Good",
    className: "good",
  })),
  ...(data.observed_actions || []).map((item, index) => ({
    id: `action-${index}`,
    type: "action",
    timestamp: toSeconds(item.timestamp_sec),
    title: item.action || "Observed action",
    body: (item.materials || []).join(", "),
    detail: item.measurement ? `Measurement: ${item.measurement}` : "",
    confidence: item.confidence,
    severity: "Action",
    className: "action",
  })),
].sort((a, b) => a.timestamp - b.timestamp || a.id.localeCompare(b.id));

const activeAnnotations = () => {
  const current = video.currentTime || 0;
  const nearby = annotations.filter(
    (item) => Math.abs(item.timestamp - current) <= ACTIVE_WINDOW_SECONDS,
  );
  if (nearby.length > 0) return nearby;

  const previous = annotations.filter((item) => item.timestamp <= current).at(-1);
  return previous ? [previous] : [];
};

const renderOverlay = (items) => {
  overlay.innerHTML = items.slice(0, 3).map((item) => `
    <article
      class="overlay-card ${item.className}"
      role="button"
      tabindex="0"
      data-time="${item.timestamp}"
      title="Jump to ${formatTime(item.timestamp)}">
      <strong>${item.title}</strong>
      <span>${item.body || item.detail || formatTime(item.timestamp)}</span>
    </article>
  `).join("");
};

const renderPanel = () => {
  if (annotations.length === 0) {
    annotationList.innerHTML = `<div class="empty">No annotations loaded.</div>`;
    return;
  }

  annotationList.innerHTML = annotations.map((item) => `
    <article
      class="annotation ${item.className}"
      role="button"
      tabindex="0"
      data-id="${item.id}"
      data-time="${item.timestamp}"
      title="Jump to ${formatTime(item.timestamp)}">
      <div class="meta">
        <span class="chip">${formatTime(item.timestamp)}</span>
        <span class="chip">${item.severity}</span>
        ${item.confidence ? `<span class="chip">${item.confidence} confidence</span>` : ""}
      </div>
      <h2>${item.title}</h2>
      ${item.body ? `<p>${item.body}</p>` : ""}
      ${item.detail ? `<p>${item.detail}</p>` : ""}
    </article>
  `).join("");
};

const renderActivePanelState = (items) => {
  activeAnnotationKeys = new Set(items.map((item) => item.id));
  annotationList.querySelectorAll("[data-id]").forEach((card) => {
    card.classList.toggle("active", activeAnnotationKeys.has(card.dataset.id));
  });
};

const seekTo = (seconds) => {
  const duration = mediaDuration();
  const target = clamp(toSeconds(seconds), 0, duration);

  if (video.readyState < 1) {
    video.addEventListener("loadedmetadata", () => seekTo(target), { once: true });
    return;
  }

  video.currentTime = target;
  render();
};

const renderTimeline = () => {
  const duration = mediaDuration();
  timeline.innerHTML = annotations.map((item) => {
    const left = Math.min(100, Math.max(0, (item.timestamp / duration) * 100));
    return `
      <button
        class="marker ${item.className}"
        style="left: ${left}%"
        title="${formatTime(item.timestamp)} ${item.title}"
        aria-label="${formatTime(item.timestamp)} ${item.title}"
        data-time="${item.timestamp}">
      </button>
    `;
  }).join("");
};

const render = () => {
  timecode.textContent = formatTime(video.currentTime || 0);
  const active = activeAnnotations();
  renderOverlay(active);
  renderActivePanelState(active);
};

const loadAnnotations = async (source) => {
  const response = await fetch(source);
  const data = await response.json();
  annotations = normalize(data);
  videoTitle.textContent = data.protocol?.title || "Yeast transformation protocol";
  renderPanel();
  renderTimeline();
  render();
};

video.src = DEFAULT_VIDEO;
loadAnnotations(DEFAULT_ANNOTATIONS).catch(() => {
  annotations = [];
  renderPanel();
  render();
});

video.addEventListener("timeupdate", render);
video.addEventListener("loadedmetadata", () => {
  renderTimeline();
  render();
});

timeline.addEventListener("click", (event) => {
  const marker = event.target.closest("[data-time]");
  if (marker) {
    event.preventDefault();
    event.stopPropagation();
    seekTo(marker.dataset.time);
    return;
  }

  const rect = timeline.getBoundingClientRect();
  const ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
  seekTo(ratio * mediaDuration());
});

annotationList.addEventListener("click", (event) => {
  const card = event.target.closest("[data-time]");
  if (!card) return;
  event.preventDefault();
  seekTo(card.dataset.time);
});

annotationList.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const card = event.target.closest("[data-time]");
  if (!card) return;
  event.preventDefault();
  seekTo(card.dataset.time);
});

overlay.addEventListener("click", (event) => {
  const card = event.target.closest("[data-time]");
  if (!card) return;
  event.preventDefault();
  event.stopPropagation();
  seekTo(card.dataset.time);
});

overlay.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const card = event.target.closest("[data-time]");
  if (!card) return;
  event.preventDefault();
  seekTo(card.dataset.time);
});

videoFile.addEventListener("change", () => {
  const [file] = videoFile.files;
  if (!file) return;
  video.src = URL.createObjectURL(file);
  videoTitle.textContent = file.name;
});

annotationFile.addEventListener("change", async () => {
  const [file] = annotationFile.files;
  if (!file) return;
  const data = JSON.parse(await file.text());
  annotations = normalize(data);
  videoTitle.textContent = data.protocol?.title || videoTitle.textContent;
  renderPanel();
  renderTimeline();
  render();
});
