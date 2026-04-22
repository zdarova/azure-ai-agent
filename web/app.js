mermaid.initialize({ startOnLoad: false, theme: 'default', securityLevel: 'loose' });
const BASE = 'https://agent-ricoh.purpleocean-d135007c.eastus2.azurecontainerapps.io';
marked.setOptions({ breaks: true, gfm: true });
let SID = localStorage.getItem('ricoh_session') || crypto.randomUUID();
localStorage.setItem('ricoh_session', SID);

const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const btnEl = document.getElementById('btn');

// --- Utilities ---

function hl(names) {
    document.querySelectorAll('.agent-item').forEach(e => e.classList.remove('active'));
    if (!names) return;
    const list = Array.isArray(names) ? names : [names];
    list.forEach(n => {
        const e = document.querySelector(`.agent-item[data-agent="${n}"]`);
        if (e) e.classList.add('active');
    });
}

function esc(t) {
    const d = document.createElement('div');
    d.textContent = t;
    return d.innerHTML;
}

function isNearBottom() {
    return chatEl.scrollHeight - chatEl.scrollTop - chatEl.clientHeight < 80;
}

function scrollBottom(force = false) {
    if (force || isNearBottom()) chatEl.scrollTop = chatEl.scrollHeight;
}

function addMsg(role, html) {
    const d = document.createElement('div');
    d.className = `message ${role}`;
    d.innerHTML = `<div class="avatar">${role === 'user' ? 'Tu' : 'AI'}</div><div class="bubble">${html}</div>`;
    chatEl.appendChild(d);
    scrollBottom();
    return d;
}

function stars(n) {
    return '★'.repeat(Math.min(n, 5)) + '☆'.repeat(Math.max(5 - n, 0));
}

// --- Mermaid helpers ---

function initMermaidSvg(svg, container) {
    svg.style.maxWidth = 'none';
    svg.style.transform = 'scale(1)';
    svg.style.transformOrigin = 'top left';
    svg.dataset.zoom = '1';
    let drag = false, sx, sy, sl, st;
    svg.addEventListener('mousedown', e => { drag = true; sx = e.clientX; sy = e.clientY; sl = container.scrollLeft; st = container.scrollTop; svg.style.cursor = 'grabbing'; });
    document.addEventListener('mousemove', e => { if (!drag) return; container.scrollLeft = sl - (e.clientX - sx); container.scrollTop = st - (e.clientY - sy); });
    document.addEventListener('mouseup', () => { drag = false; svg.style.cursor = 'grab'; });
    container.addEventListener('wheel', e => { e.preventDefault(); const z = parseFloat(svg.dataset.zoom || 1); const nz = e.deltaY < 0 ? z * 1.15 : z / 1.15; svg.dataset.zoom = nz; svg.style.transform = `scale(${nz})`; }, { passive: false });
}

async function renderMermaidBlocks(bubble) {
    bubble.querySelectorAll('code.language-mermaid').forEach(el => {
        const p = el.parentElement;
        const wrapper = document.createElement('div');
        wrapper.innerHTML = `<div class="diagram-controls">
            <button onclick="zoomDiagram(this,1.2)">🔍+</button>
            <button onclick="zoomDiagram(this,0.8)">🔍−</button>
            <button onclick="zoomDiagram(this,'fit')">Fit</button>
            <button onclick="zoomDiagram(this,'wide')">Wide</button>
        </div>`;
        const c = document.createElement('div');
        c.className = 'mermaid';
        c.textContent = el.textContent;
        wrapper.appendChild(c);
        p.replaceWith(wrapper);
    });
    try { await mermaid.run({ nodes: bubble.querySelectorAll('.mermaid') }); } catch (e) {}
    bubble.querySelectorAll('.mermaid svg').forEach(svg => initMermaidSvg(svg, svg.parentElement));
}

function zoomDiagram(btn, factor) {
    const container = btn.closest('.diagram-controls').parentElement.querySelector('.mermaid');
    const svg = container?.querySelector('svg');
    if (!svg) return;
    if (factor === 'fit') { svg.dataset.zoom = '1'; svg.style.transform = 'scale(1)'; svg.style.maxWidth = '100%'; return; }
    if (factor === 'wide') { svg.dataset.zoom = '1'; svg.style.transform = 'scale(1)'; svg.style.maxWidth = 'none'; return; }
    const z = parseFloat(svg.dataset.zoom || 1) * factor;
    svg.dataset.zoom = z; svg.style.transform = `scale(${z})`; svg.style.maxWidth = 'none';
}

// --- Quality rendering ---

function renderQ(q) {
    const s = stars;
    const oc = q.overall >= 4 ? `overall-${q.overall}` : q.overall >= 3 ? 'overall-3' : 'overall-1';
    return `<div class="quality-bar">
    <div class="quality-header"><span>✅</span><span class="title">Quality Check · LLM-as-Judge · 9 Dimensions</span></div>
    <div class="quality-scores">
        <span class="overall-badge ${oc}">${q.overall}/5</span>
        <span class="metric"><span class="metric-label">Rilevanza</span><span class="stars">${s(q.relevance)}</span></span>
        <span class="metric"><span class="metric-label">Accuratezza</span><span class="stars">${s(q.accuracy)}</span></span>
        <span class="metric"><span class="metric-label">Completezza</span><span class="stars">${s(q.completeness)}</span></span>
        <span class="metric"><span class="metric-label">Chiarezza</span><span class="stars">${s(q.clarity)}</span></span>
    </div>
    <div class="quality-scores" style="margin-top:.3rem">
        <span class="metric"><span class="metric-label">Hallucination</span><span class="stars">${s(q.hallucination)}</span></span>
        <span class="metric"><span class="metric-label">Faithfulness</span><span class="stars">${s(q.faithfulness)}</span></span>
        <span class="metric"><span class="metric-label">Ctx Precision</span><span class="stars">${s(q.context_precision)}</span></span>
        <span class="metric"><span class="metric-label">Ctx Recall</span><span class="stars">${s(q.context_recall)}</span></span>
        <span class="metric"><span class="metric-label">Toxicity/Bias</span><span class="stars">${s(q.toxicity_bias)}</span></span>
    </div>
    ${q.note ? `<div class="quality-note">"${q.note}"</div>` : ''}</div>`;
}

// --- Sidebar actions ---

async function showArchitecture() {
    try {
        const r = await fetch(BASE + '/api/architecture'), d = await r.json();
        const m = addMsg('assistant', '<span class="agent-badge">📐 architecture</span> ');
        m.classList.add('wide');
        const b = m.querySelector('.bubble');
        const wrapper = document.createElement('div');
        wrapper.innerHTML = '<div class="diagram-controls"><button onclick="zoomDiagram(this,1.2)">🔍+</button><button onclick="zoomDiagram(this,0.8)">🔍−</button><button onclick="zoomDiagram(this,\'fit\')">↩ Fit</button><button onclick="zoomDiagram(this,\'wide\')">↔ Wide</button></div>';
        const c = document.createElement('div'); c.className = 'mermaid'; c.textContent = d.diagram; wrapper.appendChild(c); b.appendChild(wrapper);
        try { await mermaid.run({ nodes: [c] }); } catch (e) {}
        c.querySelectorAll('svg').forEach(svg => initMermaidSvg(svg, c));
        scrollBottom();
    } catch (e) { addMsg('assistant', `<em>Errore: ${e.message}</em>`); }
}

async function showMetrics() {
    try {
        const r = await fetch(BASE + '/api/metrics'), d = await r.json();
        const s = n => '★'.repeat(Math.round(n)) + '☆'.repeat(5 - Math.round(n));
        let html = '<span class="agent-badge">📊 metrics</span><h3>Agent Performance</h3><table><tr><th>Agent</th><th>Calls</th><th>Avg (ms)</th><th>Last (ms)</th><th>Errors</th></tr>';
        for (const [k, v] of Object.entries(d.agents || {})) html += `<tr><td>${k}</td><td>${v.calls}</td><td>${v.avg_ms}</td><td>${v.last_ms}</td><td>${v.errors}</td></tr>`;
        html += '</table>';
        if (d.quality && d.quality.total_evals) {
            const q = d.quality;
            html += `<br><h3>Quality Averages (${q.total_evals} evaluations)</h3><table><tr><th>Metric</th><th>Avg</th><th>Rating</th></tr>`;
            const labels = { relevance: 'Rilevanza', accuracy: 'Accuratezza', completeness: 'Completezza', clarity: 'Chiarezza', hallucination: 'Hallucination ↑', faithfulness: 'Faithfulness', context_precision: 'Context Precision', context_recall: 'Context Recall', toxicity_bias: 'Toxicity/Bias ↑', overall: 'Overall' };
            for (const [k, label] of Object.entries(labels)) { if (q[k]) html += `<tr><td>${label}</td><td>${q[k]}</td><td>${s(q[k])}</td></tr>`; }
            html += '</table>';
        }
        if (d.feedback) html += `<br><strong>Feedback:</strong> 👍 ${d.feedback.thumbs_up} · 👎 ${d.feedback.thumbs_down}`;
        addMsg('assistant', html);
    } catch (e) { addMsg('assistant', `<em>Errore: ${e.message}</em>`); }
}

// --- Feedback ---

async function sendFb(mid, rating, btn) {
    btn.classList.add('sel');
    btn.parentElement.querySelectorAll('.fb-btn').forEach(b => { if (b !== btn) b.style.opacity = '.3'; });
    try { await fetch(BASE + '/api/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: SID, message_id: mid, rating }) }); } catch (e) {}
}

// --- Main chat with multi-route progressive rendering ---

async function send() {
    const q = inputEl.value.trim();
    if (!q) return;
    addMsg('user', esc(q));
    inputEl.value = '';
    btnEl.disabled = true;
    hl(null);
    scrollBottom(true);

    const msg = addMsg('assistant', '<span class="typing">⏳ Guardrails check...</span>');
    const bub = msg.querySelector('.bubble');

    try {
        const res = await fetch(BASE + '/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: q, session_id: SID }),
        });
        const reader = res.body.getReader(), dec = new TextDecoder();
        let buf = '', mid = '';
        let routes = [];
        let reasonHtml = '';
        let warningHtml = '';
        let agentResponses = {};

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });
            const lines = buf.split('\n');
            buf = lines.pop();
            let ev = '';

            for (const ln of lines) {
                if (ln.startsWith('event: ')) { ev = ln.slice(7).trim(); }
                else if (ln.startsWith('data: ') && ev) {
                    try {
                        const d = JSON.parse(ln.slice(6));

                        if (ev === 'warning') {
                            warningHtml += `<div class="warning-box">${esc(d.message)}</div>`;
                        }
                        else if (ev === 'routing') {
                            mid = d.message_id || '';
                            routes = d.agents || [d.agent];
                            const isMulti = routes.length > 1;

                            hl('guardrails');
                            setTimeout(() => hl(routes), 300);

                            if (d.reasoning) {
                                reasonHtml = `<div class="reasoning-box"><div class="label">🧠 Reasoning</div><div class="text">${esc(d.reasoning)}</div></div>`;
                            }

                            const badges = routes.map(r => `<span class="agent-badge">🤖 ${r}</span>`).join(' ');
                            const multiLabel = isMulti ? `<div style="font-size:.7rem;color:var(--muted);margin-bottom:.4rem">📡 Multi-route: ${routes.length} agents</div>` : '';
                            bub.innerHTML = warningHtml + badges + multiLabel + reasonHtml + '<span class="typing">✍️ Generating response...</span>';
                            scrollBottom();
                        }
                        else if (ev === 'agent_response') {
                            // Progressive: render each agent's response as it arrives
                            hl(d.agent);
                            agentResponses[d.agent] = d.text;

                            const badges = routes.map(r => `<span class="agent-badge">🤖 ${r}</span>`).join(' ');
                            const multiLabel = routes.length > 1 ? `<div style="font-size:.7rem;color:var(--muted);margin-bottom:.4rem">📡 Multi-route: ${routes.length} agents</div>` : '';

                            let progressHtml = '';
                            for (const r of routes) {
                                if (agentResponses[r]) {
                                    if (routes.length > 1) {
                                        progressHtml += `<div class="agent-section" style="margin-bottom:.8rem"><div class="agent-badge" style="margin-bottom:.3rem">🤖 ${r}</div>${marked.parse(agentResponses[r])}</div>`;
                                    } else {
                                        progressHtml += marked.parse(agentResponses[r]);
                                    }
                                }
                            }

                            const pending = routes.filter(r => !agentResponses[r]);
                            const pendingHtml = pending.length ? `<span class="typing">✍️ Waiting for: ${pending.join(', ')}...</span>` : '<span class="typing">🔍 Quality check...</span>';

                            bub.innerHTML = warningHtml + badges + multiLabel + reasonHtml + progressHtml + pendingHtml;
                            if (d.text.includes('```mermaid')) msg.classList.add('wide');
                            await renderMermaidBlocks(bub);
                            scrollBottom();
                        }
                        else if (ev === 'response') {
                            // Final merged response — only used if no agent_response events arrived (backward compat)
                            if (Object.keys(agentResponses).length === 0) {
                                const badges = routes.map(r => `<span class="agent-badge">🤖 ${r}</span>`).join(' ');
                                const resp = marked.parse(d.text);
                                if (d.text.includes('```mermaid')) msg.classList.add('wide');
                                bub.innerHTML = warningHtml + badges + reasonHtml + resp + '<div id="qp" class="typing" style="margin-top:.5rem">🔍 Quality check...</div>';
                                await renderMermaidBlocks(bub);
                            } else {
                                // Replace pending spinner with quality check spinner
                                const spinner = bub.querySelector('.typing');
                                if (spinner) spinner.outerHTML = '<div id="qp" class="typing" style="margin-top:.5rem">🔍 Quality check...</div>';
                            }
                            scrollBottom();
                        }
                        else if (ev === 'quality') {
                            hl('quality_check');
                            const p = bub.querySelector('#qp') || bub.querySelector('.typing');
                            if (p) p.remove();
                            if (d.quality) {
                                const qd = document.createElement('div');
                                qd.innerHTML = renderQ(d.quality);
                                bub.appendChild(qd.firstElementChild);
                            }
                            scrollBottom();
                        }
                        else if (ev === 'trace') {
                            if (d.metrics) {
                                const items = Object.entries(d.metrics).filter(([, v]) => v.last_ms > 0).map(([k, v]) => `<span>${k}:<strong>${v.last_ms}ms</strong></span>`).join(' · ');
                                if (items) {
                                    const t = document.createElement('div');
                                    t.innerHTML = `<div class="trace-bar"><div class="trace-title">⏱️ Trace</div><div class="trace-items">${items}</div></div>`;
                                    bub.appendChild(t.firstElementChild);
                                }
                            }
                        }
                        else if (ev === 'memory') {
                            if (d.new_facts > 0) {
                                const md = document.createElement('div');
                                md.innerHTML = `<div style="margin-top:.4rem;font-size:.65rem;color:#7c3aed">🧠 ${d.new_facts} new fact${d.new_facts > 1 ? 's' : ''} memorized</div>`;
                                bub.appendChild(md.firstElementChild);
                            }
                        }
                        else if (ev === 'done') {
                            hl(null);
                            const fb = document.createElement('div');
                            fb.className = 'feedback-row';
                            fb.innerHTML = `<span class="feedback-label">Utile?</span><button class="fb-btn" onclick="sendFb('${mid}','thumbs_up',this)">👍</button><button class="fb-btn" onclick="sendFb('${mid}','thumbs_down',this)">👎</button>`;
                            bub.appendChild(fb);
                        }
                        else if (ev === 'error') {
                            bub.innerHTML = `<em>🛡️ ${esc(d.message)}</em>`;
                        }
                    } catch (e) {}
                    ev = '';
                }
            }
        }
    } catch (e) {
        bub.innerHTML = `<em>Errore: ${esc(e.message)}</em>`;
    }
    btnEl.disabled = false;
    inputEl.focus();
}
