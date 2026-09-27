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
  const LIGHTBOX_ID = "sv-lightbox";
  let lastResult = null;
  const panelCss = `
    #${PANEL_ID}{margin-top:16px;border:1px solid var(--border);border-radius:10px;background:var(--card);padding:16px;box-shadow:var(--shadow)}
    #${PANEL_ID} .sv-head{display:flex;align-items:flex-start;gap:12px;justify-content:space-between;flex-wrap:wrap}
    #${PANEL_ID} .sv-title{font-size:15.5px;font-weight:700}
    #${PANEL_ID} .sv-sub{font-size:12px;color:var(--muted);margin-top:2px}
    #${PANEL_ID} .sv-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px;align-items:stretch}
    #${PANEL_ID} .sv-grid > div{display:flex;flex-direction:column;min-width:0}
    #${PANEL_ID} .sv-img-wrap{position:relative;flex:1;display:flex}
    #${PANEL_ID} .sv-img{width:100%;flex:1;min-height:260px;object-fit:contain;background:var(--bg);border:1px solid var(--border);border-radius:8px}
    #${PANEL_ID} .sv-zoom-btn{
      position:absolute;top:8px;right:8px;width:30px;height:30px;
      display:flex;align-items:center;justify-content:center;
      border-radius:7px;border:1px solid rgba(255,255,255,.25);
      background:rgba(15,15,20,.55);color:#fff;cursor:pointer;
      font-size:15px;line-height:1;padding:0;backdrop-filter:blur(3px);
      transition:background .15s ease,transform .15s ease;
    }
    #${PANEL_ID} .sv-zoom-btn:hover{background:rgba(15,15,20,.8);transform:scale(1.06)}
    #${PANEL_ID} .sv-zoom-btn:focus-visible{outline:2px solid var(--primary);outline-offset:2px}
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
    @media(max-width:720px){#${PANEL_ID} .sv-grid{grid-template-columns:1fr}#${PANEL_ID} .sv-img{min-height:200px}#${PANEL_ID} .sv-metrics{grid-template-columns:1fr 1fr}}

    #${LIGHTBOX_ID}{
      position:fixed;inset:0;z-index:9999;display:none;
      align-items:center;justify-content:center;
      background:rgba(0,0,0,.82);padding:32px;
    }
    #${LIGHTBOX_ID}.sv-open{display:flex}
    #${LIGHTBOX_ID} img{
      max-width:100%;max-height:100%;object-fit:contain;
      border-radius:8px;box-shadow:0 10px 40px rgba(0,0,0,.5);
    }
    #${LIGHTBOX_ID} .sv-lightbox-close{
      position:fixed;top:18px;right:22px;width:40px;height:40px;
      border-radius:50%;border:1px solid rgba(255,255,255,.3);
      background:rgba(20,20,24,.6);color:#fff;font-size:20px;
      display:flex;align-items:center;justify-content:center;cursor:pointer;
    }
    #${LIGHTBOX_ID} .sv-lightbox-close:hover{background:rgba(20,20,24,.9)}
    #${LIGHTBOX_ID} .sv-lightbox-caption{
      position:fixed;bottom:20px;left:0;right:0;text-align:center;
      color:#eee;font-size:12.5px;
    }
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

  // ---------- Lightbox (maximize) ----------

  function getLightbox() {
    let box = document.getElementById(LIGHTBOX_ID);
    if (box) return box;

    box = document.createElement("div");
    box.id = LIGHTBOX_ID;
    box.innerHTML = `
      <button type="button" class="sv-lightbox-close" aria-label="Close">&times;</button>
      <img alt="" />
      <div class="sv-lightbox-caption"></div>
    `;
    document.body.appendChild(box);

    const close = () => box.classList.remove("sv-open");
    box.querySelector(".sv-lightbox-close").addEventListener("click", close);
    // Click on the dark backdrop (but not the image itself) closes it
    box.addEventListener("click", (e) => {
      if (e.target === box) close();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && box.classList.contains("sv-open")) close();
    });

    return box;
  }

  function openLightbox(src, caption) {
    if (!src) return;
    const box = getLightbox();
    const img = box.querySelector("img");
    const cap = box.querySelector(".sv-lightbox-caption");
    img.src = src;
    img.alt = caption || "";
    cap.textContent = caption || "";
    box.classList.add("sv-open");
  }

  // Wraps an <img>/placeholder markup string with a maximize button.
  // `src` may be empty/undefined, in which case no button is added.
  function withMaximize(innerHtml, src, caption) {
    const btn = src
      ? `<button type="button" class="sv-zoom-btn" data-full="${esc(src)}" data-caption="${esc(caption)}" aria-label="Maximize ${esc(caption)}" title="Maximize">&#10021;</button>`
      : "";
    return `<div class="sv-img-wrap">${innerHtml}${btn}</div>`;
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

    const alignedInner = visuals.aligned
      ? `<img class="sv-img" src="${visuals.aligned}" alt="Aligned document" />`
      : `<div class="sv-img sv-muted" style="display:grid;place-items:center">No aligned preview</div>`;
    const matchesInner = visuals.match_visualization
      ? `<img class="sv-img" src="${visuals.match_visualization}" alt="Feature matches" />`
      : `<div class="sv-img sv-muted" style="display:grid;place-items:center">No match visualization</div>`;

    const aligned = withMaximize(alignedInner, visuals.aligned, "Aligned document");
    const matches = withMaximize(matchesInner, visuals.match_visualization, "Feature matches");

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

    // Wire up the maximize buttons for this render pass.
    panel.querySelectorAll(".sv-zoom-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        openLightbox(btn.getAttribute("data-full"), btn.getAttribute("data-caption"));
      });
    });
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