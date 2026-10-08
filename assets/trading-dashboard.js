"use strict";

const endpoints = {
  status: "/dashboard/trading/api/status",
  logs: "/dashboard/trading/api/logs?limit=100",
  access: "/dashboard/trading/api/access",
};
const names = ["status", "logs"];
const byId = (id) => document.getElementById(id);
let refreshTimer;

function setPanel(name, value) {
  const panel = byId(`${name}-data`);
  panel.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function setBusy(busy) {
  byId("refresh").disabled = busy;
  for (const id of ["pause", "resume", "evaluate"]) byId(id).disabled = busy;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    credentials: "same-origin",
    cache: "no-store",
    headers: { Accept: "application/json", ...(options.headers || {}) },
    ...options,
    signal: AbortSignal.timeout(8000),
  });
  let payload;
  try { payload = await response.json(); }
  catch { payload = { error: "invalid_response" }; }
  if (response.status === 401 && payload.login_url) {
    const link = byId("login-link");
    try {
      const target = new URL(payload.login_url, window.location.origin);
      if (target.protocol === "https:" && target.origin === "https://api.neuralmimicry.ai") link.href = target.href;
    } catch { /* Keep the configured safe Customers login URL. */ }
    byId("login-panel").hidden = false;
  }
  if (!response.ok) {
    const error = new Error(payload.message || payload.error || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

function showAccessError(error) {
  if (error.status === 401) {
    byId("session-state").textContent = "Sign-in required";
    return;
  }
  byId("access-error-message").textContent = error.message || "Please retry shortly.";
  byId("access-error").hidden = false;
  byId("session-state").textContent = "Access check failed";
}

async function refreshData() {
  byId("access-error").hidden = true;
  setBusy(true);
  try {
    const access = await requestJson(endpoints.access);
    byId("login-panel").hidden = true;
    byId("session-state").textContent = `Signed in as ${access.user}`;
    byId("controls").hidden = !access.can_control;
    const results = await Promise.allSettled(names.map((name) => requestJson(endpoints[name])));
    results.forEach((result, index) => {
      const name = names[index];
      if (result.status === "fulfilled") setPanel(name, result.value);
      else setPanel(name, result.reason.message || "Data is unavailable.");
    });
    byId("last-updated").textContent = `Last checked ${new Date().toLocaleString("en-GB")}. Data is retrieved directly from Gail.`;
  } catch (error) {
    showAccessError(error);
    if (error.status === 401) names.forEach((name) => setPanel(name, "Sign in to view this information."));
  } finally {
    setBusy(false);
  }
}

async function runAction(name, confirmation) {
  if (confirmation && !window.confirm(confirmation)) return;
  byId("action-result").textContent = "Submitting request…";
  setBusy(true);
  try {
    const result = await requestJson(`/dashboard/trading/api/${name}`, { method: "POST" });
    byId("action-result").textContent = result.message || "Request accepted by Gail.";
    await refreshData();
  } catch (error) {
    byId("action-result").textContent = error.message || "Gail could not complete the request.";
  } finally {
    setBusy(false);
  }
}

byId("refresh").addEventListener("click", refreshData);
byId("pause").addEventListener("click", () => runAction("pause", "Pause Gail's trading bridge?"));
byId("resume").addEventListener("click", () => runAction("resume", "Resume Gail's trading bridge?"));
byId("evaluate").addEventListener("click", () => runAction("evaluate"));
refreshData();
refreshTimer = window.setInterval(refreshData, 30000);
window.addEventListener("pagehide", () => window.clearInterval(refreshTimer), { once: true });
