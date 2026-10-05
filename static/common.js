/* Shared helpers for every page: API access and the data-status banner. */

// Pages are served by the application itself, so requests go to the same address.
const API_BASE = (location.protocol === "file:") ? "http://localhost:8000" : "";

function esc(text) {
  return String(text).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function errorFrom(data, fallback) {
  const d = data && data.detail;
  if (typeof d === "string") return { message: d, code: null, detail: null };
  if (Array.isArray(d)) {
    return { message: "Some of the values entered are not valid. Please check them and try again.", code: "INVALID_INPUT", detail: null };
  }
  if (d && typeof d === "object") return { message: d.message || fallback, code: d.code || null, detail: d };
  return { message: fallback, code: null, detail: null };
}

async function apiRequest(path, options) {
  let res;
  try {
    res = await fetch(API_BASE + path, options || {});
  } catch (e) {
    const err = new Error("Cannot reach the prediction service. Start it with start_kotmale.bat, then open http://localhost:8000.");
    err.code = "UNREACHABLE";
    throw err;
  }
  let data = null;
  try { data = await res.json(); } catch (e) { /* response had no body */ }
  if (!res.ok) {
    const info = errorFrom(data, "The request failed (" + res.status + ").");
    const err = new Error(info.message);
    err.code = info.code; err.detail = info.detail; err.status = res.status;
    throw err;
  }
  return data;
}

function apiGet(path) { return apiRequest(path); }

function apiSend(path, method, body, extraHeaders) {
  return apiRequest(path, {
    method: method,
    headers: Object.assign({ "Content-Type": "application/json" }, extraHeaders || {}),
    body: JSON.stringify(body),
  });
}

/* ------------------------------------------------------------ status banner */
let _statusPromise = null;

/* The banner reports the live data state, so it belongs on the pages that act on
   that state. The overview, catchments and model pages are read-only, and carry
   their own inline notices where they need one. */
const BANNER_PAGES = ["predict.html", "entry.html"];

function bannerShown() {
  const page = location.pathname.split("/").pop() || "index.html";
  return BANNER_PAGES.includes(page);
}

function getStatus(force) {
  if (force || !_statusPromise) {
    _statusPromise = apiGet("/status")
      .then((s) => { renderBanner(s, null); return s; })
      .catch((e) => { renderBanner(null, e.message); throw e; });
  }
  return _statusPromise;
}

function renderBanner(status, errorMessage) {
  if (!bannerShown()) return;
  let el = document.getElementById("statusBanner");
  if (!el) {
    el = document.createElement("div");
    el.id = "statusBanner";
    const nav = document.querySelector("nav.topnav");
    if (nav) nav.insertAdjacentElement("afterend", el); else document.body.prepend(el);
  }
  el.innerHTML = "";

  let level, text;
  const extra = [];
  if (errorMessage) {
    level = "error"; text = errorMessage;
  } else {
    level = status.level; text = status.message;
    if (!status.model_loaded) extra.push("Prediction model not loaded: " + (status.model_error || "unknown error"));
    if (status.system_date_overridden) extra.push("Demonstration setting: the system date is fixed at " + status.system_date + ".");
  }
  el.className = "banner banner-" + level;

  const box = document.createElement("div");
  box.className = "banner-text";
  const main = document.createElement("strong");
  main.textContent = text;
  box.appendChild(main);
  extra.forEach((t) => {
    const line = document.createElement("div");
    line.className = "banner-extra";
    line.textContent = t;
    box.appendChild(line);
  });
  el.appendChild(box);

  const onEntryPage = location.pathname.endsWith("entry.html");
  if (status && !status.ready && status.can_record && !onEntryPage) {
    const link = document.createElement("a");
    link.href = "entry.html";
    link.className = "banner-link";
    link.textContent = "Add data \u2192";
    el.appendChild(link);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  if (bannerShown()) getStatus().catch(() => {});
});
