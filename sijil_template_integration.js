/*
  Sijil UI bridge.

  Add this one line before </body> in the Sijil HTML:
      <script src="./sijil_template_integration.js"></script>

  The integrated Sijil HTML sends the image to the local verification API
  once. This script listens for the result event and inserts an explainable
  visualization card. It deliberately does not upload the image itself.
*/
(function () {
  "use strict";

  const PANEL_ID = "sijil-template-verification";
  let lastResult = null;
  const panelCss = `
    #${PANEL_ID}{margin-top:16px;border:1px solid var(--border);border-radius:10px;background:var(--card);padding:16px;box-shadow:var(--shadow)}
    #${PANEL_ID} .sv-head{display:flex;align-items:flex-start;gap:12px;justify-content:space-between;flex-wrap:wrap}
    #${PANEL_ID} .sv-title{font-size:15.5px;font-weight:700}
    #${PANEL_ID} .sv-sub{font-size:12px;color:var(--muted);margin-top:2px}
    #${PANEL_ID} .sv-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px;align-items:stretch}
    #${PANEL_ID} .sv-img{width:100%;height:150px;object-fit:contain;background:var(--bg);border:1px solid var(--border);border-radius:8px}
    #${PANEL_ID} .sv-label{font-size:11px;color:var(--muted);margin:0 0 5px}
    #${PANEL_ID} .sv-badge{display:inline-flex;align-items:center;border-radius:999px;padding:4px 10px;font-size:11.5px;font-weight:700}
    #${PANEL_ID} .sv-ok{background:var(--ok-bg);color:var(--ok)}
    #${PANEL_ID} .sv-warn{background:var(--warn-bg);color:var(--warn)}
    #${PANEL_ID} .sv-bad{background:var(--bad-bg);color:var(--bad)}
    #${PANEL_ID} .sv-muted{color:var(--muted);font-size:12.5px}
    #${PANEL_ID} .sv-metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px}
    #${PANEL_ID} .sv-metric{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:9px}
    #${PANEL_ID} .sv-metric b{display:block;font-size:16px;color:var(--primary)}
    #${PANEL_ID} .sv-metric span{font-size:10.5px;color:var(--muted)}
    #${PANEL_ID} .sv-candidates{margin-top:12px;border-top:1px solid var(--border);padding-top:10px}
    #${PANEL_ID} .sv-candidate{display:flex;justify-content:space-between;gap:10px;padding:5px 0;font-size:12px;border-bottom:1px solid var(--border)}
    #${PANEL_ID} .sv-candidate:last-child{border-bottom:0}
    #${PANEL_ID} .sv-auth{margin-top:12px;border-top:1px solid var(--border);padding-top:12px}
    #${PANEL_ID} .sv-auth-head{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}
    #${PANEL_ID} .sv-auth-bar{margin-top:8px;height:8px;border-radius:999px;background:var(--bg);border:1px solid var(--border);overflow:hidden}
    #${PANEL_ID} .sv-auth-fill{height:100%;border-radius:999px}
    #${PANEL_ID} .sv-auth-note{margin-top:6px;font-size:11px;color:var(--muted)}
    #${PANEL_ID} .sv-loading{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:13px}
    #${PANEL_ID} .sv-dot{width:8px;height:8px;background:var(--primary);border-radius:50%;animation:svpulse 1s infinite}
    @keyframes svpulse{0%,100%{opacity:.3}50%{opacity:1}}
    @media(max-width:720px){#${PANEL_ID} .sv-grid{grid-template-columns:1fr}#${PANEL_ID} .sv-metrics{grid-template-columns:1fr 1fr}}
  `;

  function esc(value) {
    return String(value ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    }[c]));
  }

  function injectCss() {
    if (document.getElementById("sijil-template-integration-css")) return;
    const style = document.createElement("style");
    style.id = "sijil-template-integration-css";
    style.textContent = panelCss;
    document.head.appendChild(style);
  }

  function statusClass(verdict) {
    if (String(verdict).includes("CONFIRMED")) return "sv-ok";
    if (String(verdict).includes("WEAK") || String(verdict).includes("NO_")) return "sv-bad";
    return "sv-warn";
  }

  function authClass(verdict) {
    if (verdict === "LIKELY_GENUINE") return "sv-ok";
    if (verdict === "LIKELY_FAKE") return "sv-bad";
    return "sv-warn";
  }

  function authBarColor(verdict) {
    if (verdict === "LIKELY_GENUINE") return "var(--ok)";
    if (verdict === "LIKELY_FAKE") return "var(--bad)";
    return "var(--warn)";
  }

  function renderAuthenticity(authenticity) {
    if (!authenticity) return "";
    if (!authenticity.available) {
      return `
        <div class="sv-auth">
          <div class="sv-auth-head">
            <span class="sv-label" style="margin:0">Authenticity check</span>
            <span class="sv-badge sv-muted">Unavailable</span>
          </div>
          <div class="sv-auth-note">${esc(authenticity.error || "Authenticity model not loaded on the backend.")}</div>
        </div>
      `;
    }
    const genuinePct = Number(authenticity.genuine_percent || 0);
    const cls = authClass(authenticity.verdict);
    const barColor = authBarColor(authenticity.verdict);
    const verdictLabel = String(authenticity.verdict || "").replace(/_/g, " ");
    return `
      <div class="sv-auth">
        <div class="sv-auth-head">
          <span class="sv-label" style="margin:0">Authenticity check</span>
          <span class="sv-badge ${cls}">${esc(verdictLabel)}</span>
        </div>
        <div class="sv-muted" style="margin-top:6px">
          Genuine probability: <b>${genuinePct.toFixed(1)}%</b> · Fake/tampered probability: <b>${(100 - genuinePct).toFixed(1)}%</b>
        </div>
        <div class="sv-auth-bar"><div class="sv-auth-fill" style="width:${genuinePct}%;background:${barColor}"></div></div>
        <div class="sv-auth-note">${esc(authenticity.note || "")}</div>
      </div>
    `;
  }

  function createPanel() {
    let panel = document.getElementById(PANEL_ID);
    if (panel) return panel;
    const drop = document.querySelector(".drop");
    if (!drop) return null;
    panel = document.createElement("section");
    panel.id = PANEL_ID;
    panel.innerHTML = `
      <div class="sv-head">
        <div>
          <div class="sv-title">Template verification</div>
          <div class="sv-sub">OCR remains unchanged; this is an additional structural check.</div>
        </div>
        <span class="sv-badge sv-muted" id="sv-status">Waiting for image</span>
      </div>
      <div id="sv-body" class="sv-muted" style="margin-top:12px">
        Upload a document to compare it with the reference templates.
      </div>
    `;
    drop.insertAdjacentElement("afterend", panel);
    // This app fully rebuilds the drop zone on every internal state change
    // (view.innerHTML = ...), which destroys this panel along with it since
    // it was inserted as a DOM sibling. If we have a cached result from
    // before the rebuild, restore it immediately instead of showing the
    // empty placeholder - this is what keeps the panel from appearing to
    // "lose" sections (like authenticity) after the app re-renders.
    if (lastResult) {
      renderResult(lastResult);
    }
    return panel;
  }

  function renderLoading() {
    lastResult = null;
    const panel = createPanel();
    if (!panel) return;
    panel.querySelector("#sv-status").className = "sv-badge sv-warn";
    panel.querySelector("#sv-status").textContent = "Checking…";
    panel.querySelector("#sv-body").innerHTML =
      '<div class="sv-loading"><span class="sv-dot"></span>Running visual template verification…</div>';
  }

  function renderError(message) {
    const panel = createPanel();
    if (!panel) return;
    panel.querySelector("#sv-status").className = "sv-badge sv-bad";
    panel.querySelector("#sv-status").textContent = "Unavailable";
    panel.querySelector("#sv-body").innerHTML =
      `<div class="sv-muted">${esc(message)}</div>
       <div class="note" style="margin-top:10px">The OCR flow is still available. Start the local Sijil verification backend on port 8787.</div>`;
  }

  function renderResult(result) {
    const panel = createPanel();
    if (!panel) return;

    const decision = result.integration_decision || {};
    const match = result.template_match || {};
    const best = match.best || {};
    const ocr = result.ocr || {};
    const record = ocr.record || {};
    const verdict = decision.verdict || match.status || "REVIEW";
    const cls = statusClass(verdict);
    const top = match.top_candidates || [];
    const visuals = result.visuals || {};

    const bestName = best.name || "No confident template";
    const type = record.document_type || "Not identified by OCR";
    const country = record.issuing_country || "Not identified by OCR";
    const aligned = visuals.aligned
      ? `<img class="sv-img" src="${visuals.aligned}" alt="Aligned document" />`
      : `<div class="sv-img sv-muted" style="display:grid;place-items:center">No aligned preview</div>`;
    const matches = visuals.match_visualization
      ? `<img class="sv-img" src="${visuals.match_visualization}" alt="Feature matches" />`
      : `<div class="sv-img sv-muted" style="display:grid;place-items:center">No match visualization</div>`;
    const template = visuals.template_preview
      ? `<img class="sv-img" src="${visuals.template_preview}" alt="Best template" />`
      : `<div class="sv-img sv-muted" style="display:grid;place-items:center">No template preview</div>`;

    const candidateRows = top.map((item) => `
      <div class="sv-candidate">
        <span>${esc(item.name)}</span>
        <span class="mono">${Number(item.score || 0).toFixed(3)} · ${item.inliers || 0} inliers</span>
      </div>
    `).join("");

    const authenticityHtml = renderAuthenticity(result.authenticity);

    panel.querySelector("#sv-status").className = `sv-badge ${cls}`;
    panel.querySelector("#sv-status").textContent = verdict;
    panel.querySelector("#sv-body").innerHTML = `
      <div class="sv-muted">
        OCR type: <b>${esc(type)}</b> · OCR country: <b>${esc(country)}</b>
      </div>
      <div class="sv-metrics">
        <div class="sv-metric"><b>${esc(bestName)}</b><span>Best template</span></div>
        <div class="sv-metric"><b>${Number(best.score || 0).toFixed(3)}</b><span>Geometry score</span></div>
        <div class="sv-metric"><b>${best.inliers || 0}</b><span>RANSAC inliers</span></div>
        <div class="sv-metric"><b>${best.inlier_ratio ? (Number(best.inlier_ratio) * 100).toFixed(1) + "%" : "—"}</b><span>Inlier ratio</span></div>
      </div>
      <div class="sv-grid">
        <div><div class="sv-label">Aligned document</div>${aligned}</div>
        <div><div class="sv-label">Feature matches</div>${matches}</div>
      </div>
      ${candidateRows ? `<div class="sv-candidates"><div class="sv-label">Top candidates</div>${candidateRows}</div>` : ""}
      ${authenticityHtml}
    `;
  }

  function scan() {
    injectCss();
    createPanel();
  }

  window.addEventListener("sijil-verification-start", renderLoading);
  window.addEventListener("sijil-verification-result", (event) => {
    lastResult = event.detail || {};
    renderResult(lastResult);
  });
  window.addEventListener("sijil-verification-error", (event) => {
    const detail = event.detail || {};
    renderError(detail.message || "Verification backend unavailable.");
  });

  const observer = new MutationObserver(scan);
  observer.observe(document.body, { childList: true, subtree: true });
  window.addEventListener("DOMContentLoaded", scan);
  scan();
})();