"use strict";

(() => {
  let adminToken = "";
  const status = (id, message, kind = "") => {
    const element = document.getElementById(id);
    element.textContent = message;
    element.className = `status ${kind}`.trim();
  };
  const request = async (path, options = {}) => {
    if (!adminToken) throw new Error("Connect with a model-admin token first.");
    const headers = new Headers(options.headers || {});
    headers.set("Authorization", `Bearer ${adminToken}`);
    if (options.body) headers.set("Content-Type", "application/json");
    const response = await fetch(path, { ...options, headers, cache: "no-store" });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `Request failed (${response.status}).`);
    return payload;
  };
  const renderProviders = (providers) => {
    const list = document.getElementById("provider-list");
    list.replaceChildren();
    if (!providers.length) {
      const empty = document.createElement("li");
      empty.textContent = "No provider credentials configured.";
      list.append(empty);
      return;
    }
    for (const provider of providers) {
      const item = document.createElement("li");
      const details = document.createElement("div");
      const name = document.createElement("strong");
      name.textContent = provider.provider;
      const kinds = document.createElement("span");
      const configured = [];
      if (provider.username_configured) configured.push("username");
      if (provider.password_configured) configured.push("password");
      if (provider.token_configured) configured.push("token");
      kinds.textContent = `Configured: ${configured.join(", ") || "none"}`;
      details.append(name, kinds);
      const revoke = document.createElement("button");
      revoke.type = "button";
      revoke.className = "danger";
      revoke.textContent = "Revoke";
      revoke.addEventListener("click", async () => {
        if (!window.confirm(`Revoke ${provider.provider} credentials?`)) return;
        try {
          await request(`/v1/admin/model-provider-credentials?provider=${encodeURIComponent(provider.provider)}`, { method: "DELETE" });
          status("credential-status", `${provider.provider} credentials revoked.`, "success");
          await refreshProviders();
        } catch (error) {
          status("credential-status", error.message, "error");
        }
      });
      item.append(details, revoke);
      list.append(item);
    }
  };
  const refreshProviders = async () => {
    const payload = await request("/v1/admin/model-provider-credentials");
    renderProviders(payload.providers || []);
  };

  document.getElementById("connect-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    adminToken = document.getElementById("admin-token").value;
    document.getElementById("admin-token").value = "";
    try {
      await refreshProviders();
      status("connection-status", "Connected. The administrator token is held only in page memory.", "success");
    } catch (error) {
      adminToken = "";
      status("connection-status", error.message, "error");
    }
  });

  document.getElementById("credential-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const fields = {
      provider: document.getElementById("provider").value.trim(),
      username: document.getElementById("username").value,
      password: document.getElementById("password").value,
      token: document.getElementById("provider-token").value,
    };
    try {
      await request("/v1/admin/model-provider-credentials", { method: "POST", body: JSON.stringify(fields) });
      status("credential-status", `${fields.provider} credentials saved. The secret values were not returned.`, "success");
      document.getElementById("username").value = "";
      document.getElementById("password").value = "";
      document.getElementById("provider-token").value = "";
      await refreshProviders();
    } catch (error) {
      status("credential-status", error.message, "error");
    } finally {
      document.getElementById("username").value = "";
      document.getElementById("password").value = "";
      document.getElementById("provider-token").value = "";
      fields.username = "";
      fields.password = "";
      fields.token = "";
    }
  });

  document.getElementById("refresh-button").addEventListener("click", async () => {
    try {
      await refreshProviders();
      status("credential-status", "Provider status refreshed.", "success");
    } catch (error) {
      status("credential-status", error.message, "error");
    }
  });
  document.getElementById("refresh-list").addEventListener("click", async () => {
    try {
      await refreshProviders();
    } catch (error) {
      status("credential-status", error.message, "error");
    }
  });
})();
