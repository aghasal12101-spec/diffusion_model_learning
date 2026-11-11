// 我的学习：阅读状态、笔记、疑问、自测掌握情况、研究想法，以及论文传承图。
// 数据保存在 artifact 的 db（data/users/<id>/，仅本人可见，跨设备同步）；不可用时退回浏览器本地存储。
(function () {
  const QUIZ = window.PAPER_QUIZ || {};
  const STATUS = [["unread", "未读"], ["reading", "在读"], ["understood", "读懂"], ["reproduced", "已复现"]];
  const SLABEL = Object.fromEntries(STATUS);
  const LS_KEY = "fsg-learning-v1";
  const QUIZ_TOTAL = Object.values(QUIZ).reduce((n, a) => n + a.length, 0);

  const S = { papers: {}, ideas: [], sync: "local", syncDetail: "" };
  let col = null;
  const writing = {}, dirty = {}, timers = {}, pending = new Set();

  /* ---------- storage ---------- */
  function lsLoad() {
    try {
      const j = JSON.parse(localStorage.getItem(LS_KEY) || "null");
      if (j) { S.papers = j.papers || {}; S.ideas = j.ideas || []; }
    } catch (e) {}
  }
  function lsSave() {
    try { localStorage.setItem(LS_KEY, JSON.stringify({ papers: S.papers, ideas: S.ideas })); } catch (e) {}
  }
  function bodyFor(id) {
    if (id === "ideas") return { items: S.ideas, updatedAt: Date.now() };
    const k = id.slice(2);
    return { ...(S.papers[k] || {}), k };
  }
  async function writeDoc(id) {
    if (!col) { pending.delete(id); return; }
    if (writing[id]) { dirty[id] = true; return; }
    writing[id] = true;
    try {
      do { dirty[id] = false; await col.doc(id).set(bodyFor(id)); } while (dirty[id]);
      pending.delete(id);
      if (S.sync !== "cloud") setSync("cloud");
    } catch (e) {
      setSync("error", e && e.code === "quota_exceeded" ? "云端存储已满，请删除一些想法后再试。" : "");
    } finally { writing[id] = false; }
  }
  function schedule(id, ms) {
    lsSave();
    pending.add(id);
    clearTimeout(timers[id]);
    timers[id] = setTimeout(() => writeDoc(id), ms);
  }

  const getP = k => S.papers[k] || {};
  function setP(k, patch, ms) {
    S.papers[k] = { ...getP(k), ...patch, updatedAt: Date.now() };
    schedule("p_" + k, ms == null ? 500 : ms);
    refresh({ keepDrawerInputs: true });
  }
  function setIdeas(list) {
    S.ideas = list;
    schedule("ideas", 200);
    refresh({ keepDrawerInputs: true });
  }
  window.statusOf = k => getP(k).status || "unread";

  function setSync(mode, detail) {
    S.sync = mode; S.syncDetail = detail || "";
    const el = document.getElementById("meSync");
    if (!el) return;
    const text = {
      cloud: "已同步到云端 · 仅你自己可见 · 换设备也能看到",
      connecting: "正在连接云端…",
      slow: "暂时连不上云端，修改先保存在这个浏览器里，连上后会自动同步",
      local: "当前环境没有云端存储，记录只保存在这个浏览器里",
      error: "保存到云端失败，修改已先保存在这个浏览器里"
    }[mode];
    el.textContent = text + (S.syncDetail ? " " + S.syncDetail : "");
    el.className = "me-sync " + mode;
  }

  async function initCloud() {
    const c = window.claude;
    if (!c || !c.use) { setSync("local"); return; }
    let db = null, user = null;
    try { [db, user] = await Promise.all([c.use("db"), c.use("user")]); } catch (e) {}
    if (!db || !user) { setSync("local"); return; }
    let uid = null;
    try { uid = await user.id(); } catch (e) {}
    if (!uid) { setSync("local"); return; }
    try { col = db.collection("data/users/" + uid); } catch (e) { setSync("local"); return; }
    setSync("connecting");
    const t = setTimeout(() => { if (S.sync === "connecting") setSync("slow"); }, 10000);
    let confirmed = false;
    col.onSnapshot(snap => {
      const papers = {}; let ideas = null;
      snap.docs.forEach(d => {
        const v = d.data() || {};
        if (d.id === "ideas") ideas = Array.isArray(v.items) ? v.items : [];
        else if (d.id.startsWith("p_")) papers[d.id.slice(2)] = v;
      });
      if (!snap.metadata.fromCache && !confirmed) {
        confirmed = true; clearTimeout(t);
        // 第一次连上云端且云端为空：把本浏览器里已有的记录搬上去
        if (snap.empty && (Object.keys(S.papers).length || S.ideas.length)) {
          Object.keys(S.papers).forEach(k => { pending.add("p_" + k); writeDoc("p_" + k); });
          if (S.ideas.length) { pending.add("ideas"); writeDoc("ideas"); }
          setSync("cloud");
          return;
        }
        setSync("cloud");
      }
      const merged = { ...papers };
      pending.forEach(id => { if (id.startsWith("p_") && S.papers[id.slice(2)]) merged[id.slice(2)] = S.papers[id.slice(2)]; });
      S.papers = merged;
      if (ideas && !pending.has("ideas")) S.ideas = ideas;
      lsSave();
      refresh({ keepDrawerInputs: true });
    }, err => {
      clearTimeout(t);
      setSync("error");
    });
  }

  /* ---------- helpers ---------- */
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const paperOf = k => PAPERS.find(p => p.k === k);
  function problemTitles() {
    return [...document.querySelectorAll("#problems .prob h3")].map(h => {
      const c = h.cloneNode(true); c.querySelectorAll(".heat").forEach(x => x.remove()); return c.textContent.trim();
    });
  }
  function masteredCount() {
    return Object.keys(QUIZ).reduce((n, k) => n + ((getP(k).mastered || []).filter(i => i < QUIZ[k].length).length), 0);
  }
  function fmtDate(ms) {
    const d = new Date(ms); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  }

  /* ---------- dashboard ---------- */
  function renderDashboard() {
    const counts = { unread: 0, reading: 0, understood: 0, reproduced: 0 };
    PAPERS.forEach(p => { counts[window.statusOf(p.k)]++; });
    const total = PAPERS.length;
    const bar = document.getElementById("meBar");
    bar.innerHTML = "";
    ["reproduced", "understood", "reading"].forEach(s => {
      if (!counts[s]) return;
      const seg = el("span", "me-seg " + s);
      seg.style.width = (counts[s] / total * 100).toFixed(2) + "%";
      seg.title = `${SLABEL[s]} ${counts[s]}`;
      bar.appendChild(seg);
    });
    const stats = document.getElementById("meStats");
    stats.innerHTML = "";
    STATUS.forEach(([s, label]) => {
      const it = el("div", "me-stat");
      it.appendChild(el("span", "me-dot " + s));
      it.appendChild(el("span", "me-stat-label", label));
      it.appendChild(el("b", "mono", String(counts[s])));
      stats.appendChild(it);
    });
    const q = el("div", "me-stat");
    q.appendChild(el("span", "me-stat-label", "自测掌握"));
    q.appendChild(el("b", "mono", `${masteredCount()} / ${QUIZ_TOTAL}`));
    stats.appendChild(q);

    const reading = document.getElementById("meReading");
    reading.innerHTML = "";
    const rl = PAPERS.filter(p => window.statusOf(p.k) === "reading");
    if (!rl.length) reading.appendChild(el("p", "me-empty", "还没有标记为「在读」的论文。打开任意论文的解读，在顶部选择「在读」。"));
    rl.forEach(p => {
      const b = el("button", "me-chip", p.n); b.type = "button"; b.dataset.k = p.k; reading.appendChild(b);
    });

    const notes = document.getElementById("meNotes");
    notes.innerHTML = "";
    const nl = PAPERS.map(p => ({ p, v: getP(p.k) }))
      .filter(x => (x.v.note && x.v.note.trim()) || (x.v.question && x.v.question.trim()))
      .sort((a, b) => (b.v.updatedAt || 0) - (a.v.updatedAt || 0)).slice(0, 5);
    if (!nl.length) notes.appendChild(el("p", "me-empty", "笔记和疑问会出现在这里。在解读面板顶部写下你的理解或没想通的地方。"));
    nl.forEach(({ p, v }) => {
      const b = el("button", "me-note"); b.type = "button"; b.dataset.k = p.k;
      const top = el("span", "me-note-top");
      top.appendChild(el("b", null, p.n));
      if (v.question && v.question.trim()) top.appendChild(el("span", "me-q-flag", "有疑问"));
      b.appendChild(top);
      const txt = (v.note && v.note.trim()) || v.question.trim();
      b.appendChild(el("span", "me-note-text", txt.length > 90 ? txt.slice(0, 90) + "…" : txt));
      notes.appendChild(b);
    });

    renderIdeas();
  }

  let confirmDel = null;
  function renderIdeas() {
    const list = document.getElementById("ideaList");
    const titles = problemTitles();
    list.innerHTML = "";
    if (!S.ideas.length) list.appendChild(el("p", "me-empty", "读论文时冒出的研究想法、想做的小实验，都可以记在这里，并关联到下面的某个开放问题。"));
    [...S.ideas].sort((a, b) => (b.at || 0) - (a.at || 0)).forEach(it => {
      const li = el("div", "idea");
      const meta = el("div", "idea-meta");
      meta.appendChild(el("span", "mono", fmtDate(it.at || Date.now())));
      if (it.problem) meta.appendChild(el("span", "tag domain", `问题 ${it.problem} · ${titles[it.problem - 1] || ""}`));
      const del = el("button", "idea-del", confirmDel === it.id ? "确认删除" : "删除");
      del.type = "button"; del.dataset.id = it.id;
      if (confirmDel === it.id) del.classList.add("armed");
      meta.appendChild(del);
      li.appendChild(meta);
      li.appendChild(el("p", "idea-text", it.text));
      list.appendChild(li);
    });
  }

  function wireDashboard() {
    const sel = document.getElementById("ideaProb");
    problemTitles().forEach((t, i) => {
      const o = document.createElement("option"); o.value = String(i + 1); o.textContent = `问题 ${i + 1}：${t}`; sel.appendChild(o);
    });
    document.getElementById("ideaForm").addEventListener("submit", e => {
      e.preventDefault();
      const ta = document.getElementById("ideaText");
      const text = ta.value.trim();
      if (!text) { ta.focus(); return; }
      const id = "i" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
      setIdeas([...S.ideas, { id, text, problem: Number(sel.value) || 0, at: Date.now() }]);
      ta.value = ""; sel.value = "0";
    });
    document.getElementById("ideaList").addEventListener("click", e => {
      const b = e.target.closest(".idea-del"); if (!b) return;
      if (confirmDel === b.dataset.id) { confirmDel = null; setIdeas(S.ideas.filter(x => x.id !== b.dataset.id)); }
      else { confirmDel = b.dataset.id; renderIdeas(); }
    });
    document.getElementById("me").addEventListener("click", e => {
      const b = e.target.closest("[data-k]"); if (b && !b.closest("#ideaList")) openPaper(b.dataset.k);
    });
  }

  /* ---------- cards ---------- */
  function renderCardStatus() {
    document.querySelectorAll(".p-status").forEach(span => {
      const s = window.statusOf(span.dataset.statusFor);
      span.textContent = s === "unread" ? "" : SLABEL[s];
      span.className = "p-status " + s;
      span.hidden = s === "unread";
    });
    document.querySelectorAll(".p-quiz").forEach(span => {
      const k = span.dataset.quizFor, n = (QUIZ[k] || []).length;
      const m = (getP(k).mastered || []).filter(i => i < n).length;
      span.textContent = `自测 ${m}/${n}`;
      span.classList.toggle("done", n > 0 && m === n);
    });
  }

  /* ---------- drawer ---------- */
  let openK = null;
  function learnPanel(k) {
    const v = getP(k);
    const wrap = el("div", "learn");
    const row = el("div", "learn-row");
    row.appendChild(el("span", "learn-label", "我的状态"));
    const seg = el("div", "seg"); seg.setAttribute("role", "group"); seg.setAttribute("aria-label", "阅读状态");
    STATUS.forEach(([s, label]) => {
      const b = el("button", "seg-btn " + s, label); b.type = "button"; b.dataset.s = s;
      b.setAttribute("aria-pressed", String((v.status || "unread") === s));
      seg.appendChild(b);
    });
    row.appendChild(seg);
    wrap.appendChild(row);
    const grid = el("div", "learn-grid");
    [["lnNote", "note", "笔记", "用自己的话复述核心思想、关键推导，或者和其他论文的联系…", 4],
     ["lnQ", "question", "疑问", "还没想通的地方，之后回来再看…", 2]].forEach(([id, field, label, ph, rows]) => {
      const f = el("div", "learn-field");
      const l = el("label", null, label); l.htmlFor = id;
      const ta = document.createElement("textarea"); ta.id = id; ta.rows = rows; ta.placeholder = ph; ta.value = v[field] || "";
      ta.dataset.field = field;
      f.appendChild(l); f.appendChild(ta); grid.appendChild(f);
    });
    wrap.appendChild(grid);
    wrap.appendChild(el("span", "learn-save", ""));
    return wrap;
  }

  function quizPanel(k) {
    const qs = QUIZ[k]; if (!qs) return null;
    const mastered = new Set(getP(k).mastered || []);
    const sec = el("section", "d-sec quiz");
    const h = el("h4", null, `自测题 · ${qs.length} 道`);
    sec.appendChild(h);
    sec.appendChild(el("p", "d-note", "先自己想或在纸上推一遍，再点开解答。能独立讲清楚就勾上「我能答出来」。"));
    qs.forEach((item, i) => {
      const box = el("div", "qa");
      const q = el("div", "qa-q"); q.innerHTML = `<span class="qa-n mono">Q${i + 1}</span><span>${item.q}</span>`;
      const ctr = el("div", "qa-ctl");
      const show = el("button", "qa-show", "显示解答"); show.type = "button"; show.setAttribute("aria-expanded", "false");
      const lab = el("label", "qa-done");
      const cb = document.createElement("input"); cb.type = "checkbox"; cb.id = `qa-${k}-${i}`; cb.dataset.i = String(i); cb.checked = mastered.has(i);
      lab.htmlFor = cb.id; lab.appendChild(cb); lab.appendChild(document.createTextNode(" 我能答出来"));
      ctr.appendChild(show); ctr.appendChild(lab);
      const a = el("div", "qa-a"); a.hidden = true; a.innerHTML = item.a;
      show.addEventListener("click", () => {
        a.hidden = !a.hidden; show.textContent = a.hidden ? "显示解答" : "收起解答"; show.setAttribute("aria-expanded", String(!a.hidden));
      });
      box.appendChild(q); box.appendChild(ctr); box.appendChild(a);
      sec.appendChild(box);
    });
    return sec;
  }

  window.onPaperOpen = function (k, body) {
    openK = k;
    const slot = body.querySelector("#dLearn");
    if (slot) slot.replaceWith(learnPanel(k));
    const qslot = body.querySelector("#dQuiz");
    const qp = quizPanel(k);
    if (qslot) { if (qp) qslot.replaceWith(qp); else qslot.remove(); }
  };
  window.onPaperClose = function () { openK = null; };

  function wireDrawerLearning() {
    const body = document.getElementById("dBody");
    body.addEventListener("click", e => {
      const b = e.target.closest(".seg-btn"); if (!b || !openK) return;
      setP(openK, { status: b.dataset.s }, 150);
    });
    body.addEventListener("input", e => {
      const ta = e.target.closest(".learn textarea"); if (!ta || !openK) return;
      setP(openK, { [ta.dataset.field]: ta.value }, 900);
      const s = body.querySelector(".learn-save"); if (s) s.textContent = "已自动保存";
    });
    body.addEventListener("change", e => {
      const cb = e.target.closest(".qa-done input"); if (!cb || !openK) return;
      const set = new Set(getP(openK).mastered || []);
      const i = Number(cb.dataset.i);
      if (cb.checked) set.add(i); else set.delete(i);
      setP(openK, { mastered: [...set].sort((a, b) => a - b) }, 150);
    });
  }

  function refreshDrawer() {
    if (!openK) return;
    const body = document.getElementById("dBody");
    const v = getP(openK);
    body.querySelectorAll(".seg-btn").forEach(b => b.setAttribute("aria-pressed", String((v.status || "unread") === b.dataset.s)));
    body.querySelectorAll(".learn textarea").forEach(ta => {
      if (document.activeElement !== ta && ta.value !== (v[ta.dataset.field] || "")) ta.value = v[ta.dataset.field] || "";
    });
    const m = new Set(v.mastered || []);
    body.querySelectorAll(".qa-done input").forEach(cb => { cb.checked = m.has(Number(cb.dataset.i)); });
  }

  /* ---------- lineage graph ---------- */
  const LANES = [
    ["b-reflow", "拉直轨迹"], ["b-cm", "一致性模型"], ["b-flowmap", "Flow map / 平均速度"], ["b-dist", "分布层面从头训练"],
    ["b-adv", "对抗训练"], ["b-dmd", "分布匹配蒸馏"], ["ext", "跨模态与后训练"]
  ];
  const SHORT = {
    boffi_cm: "自蒸馏框架", pmf: "pMF", temf: "TE-MeanFlow", rf_improve: "改进 RF", fmm: "Flow Map Matching", cumflow: "累积 Flow Map",
    mf_var: "MF 方差削减", emf: "Euler MF", amd: "AMD", dfd: "Data-Forcing", ddmd: "Decoupled DMD", afm: "Adv. Flow", shortcut: "Shortcut",
    cm: "CM", ayf: "Align Your Flow", drifting: "Drifting", priming: "Priming", rf: "Rectified Flow", mfm: "Meta Flow Maps",
    vfm: "Variational FM", fmgrpo: "Flow-Map GRPO", coupling: "Coupling", didi: "DiDi-Instruct", diffratio: "DiffRatio", rdm: "R_dm"
  };
  // 思想传承（整理者归纳）：from → to
  const EDGES = [
    ["rf", "instaflow"], ["rf", "rf_improve"], ["rf", "otnfm"], ["rf", "afm"],
    ["cm", "ect"], ["cm", "scm"], ["cm", "shortcut"], ["cm", "imm"], ["cm", "t3d"],
    ["scm", "rcm"], ["scm", "ayf"], ["scm", "facm"], ["facm", "decm"], ["scm", "decm"],
    ["fmm", "boffi_cm"], ["fmm", "ayf"], ["fmm", "meanflow"], ["shortcut", "meanflow"], ["shortcut", "esc"],
    ["meanflow", "imf"], ["imf", "pmf"], ["meanflow", "temf"], ["meanflow", "soflow"], ["meanflow", "emf"],
    ["meanflow", "mf_var"], ["imf", "mf_var"], ["meanflow", "cumflow"], ["meanflow", "esc"], ["meanflow", "tvm"],
    ["imm", "tvm"], ["boffi_cm", "tvm"], ["boffi_cm", "emf"],
    ["dmd", "dmd2"], ["dmd", "diffratio"], ["dmd2", "ddmd"], ["dmd2", "dmdr"], ["dmd2", "rcm"], ["dmd2", "advdmd"],
    ["dmd2", "dfd"], ["dmd2", "dymd"], ["ddmd", "amd"], ["dmdr", "amd"], ["dmdr", "rdm"], ["dmdr", "advdmd"],
    ["apt", "seedvr2"], ["apt", "afm"], ["afm", "nextgan"], ["imm", "drifting"], ["drifting", "wflow"], ["drifting", "priming"],
    ["drifting", "nextgan"], ["boffi_cm", "fmlm"], ["ayf", "fmlm"], ["dmd", "didi"], ["otnfm", "coupling"],
    ["fmm", "mfm"], ["meanflow", "vfm"], ["meanflow", "fmgrpo"], ["cm", "fmgrpo"]
  ];
  const NS = "http://www.w3.org/2000/svg";
  const svgEl = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const a in attrs) e.setAttribute(a, attrs[a]); return e; };
  const graph = { nodes: {}, edges: [] };

  function monthIndex(d) {
    const [y, m] = d.split("-").map(Number);
    return (y - 2022) * 12 + ((m || 3) - 1) - 8;
  }
  function textW(s) { let w = 0; for (const ch of s) w += /[　-鿿＀-￯]/.test(ch) ? 12.5 : 6.9; return Math.ceil(w); }

  function buildGraph() {
    const svg = document.getElementById("lineageSvg");
    const PX = 25, X0 = 170, ROW = 30, TOP = 40, PADL = 14;
    const maxM = monthIndex("2026-11");
    const W = X0 + maxM * PX + 150;
    let y = TOP;
    const laneY = [];
    LANES.forEach(([lid, lname], li) => {
      const ps = PAPERS.filter(p => p.b === lid).map(p => {
        const label = SHORT[p.k] || p.n;
        return { p, label, x: X0 + monthIndex(p.d) * PX, w: textW(label) + 20 };
      }).sort((a, b) => a.x - b.x);
      const rows = [];
      ps.forEach(n => {
        let r = rows.findIndex(right => right + 8 <= n.x);
        if (r < 0) { rows.push(0); r = rows.length - 1; }
        rows[r] = n.x + n.w;
        n.y = y + PADL + r * ROW;
        graph.nodes[n.p.k] = n;
      });
      const h = PADL * 2 + Math.max(1, rows.length) * ROW - 8;
      laneY.push([lid, lname, y, h, li]);
      y += h + 6;
    });
    const H = y + 6;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("width", W); svg.setAttribute("height", H);
    svg.innerHTML = "";
    const defs = svgEl("defs", {});
    const mk = svgEl("marker", { id: "lgArrow", viewBox: "0 0 8 8", refX: "7", refY: "4", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" });
    mk.appendChild(svgEl("path", { d: "M0,0 L8,4 L0,8 z", class: "lg-arrow" }));
    const mk2 = svgEl("marker", { id: "lgArrowHl", viewBox: "0 0 8 8", refX: "7", refY: "4", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" });
    mk2.appendChild(svgEl("path", { d: "M0,0 L8,4 L0,8 z", class: "lg-arrow-hl" }));
    defs.appendChild(mk); defs.appendChild(mk2); svg.appendChild(defs);

    // lanes
    laneY.forEach(([lid, lname, ly, lh, li]) => {
      svg.appendChild(svgEl("rect", { x: 0, y: ly, width: W, height: lh, class: li % 2 ? "lg-lane odd" : "lg-lane" }));
      const t = svgEl("text", { x: 12, y: ly + lh / 2 + 4, class: "lg-lane-label" }); t.textContent = lname; svg.appendChild(t);
    });
    // years
    for (let yr = 2023; yr <= 2026; yr++) {
      const x = X0 + monthIndex(`${yr}-01`) * PX;
      svg.appendChild(svgEl("line", { x1: x, y1: TOP - 10, x2: x, y2: H - 4, class: "lg-year-line" }));
      const t = svgEl("text", { x: x + 4, y: TOP - 16, class: "lg-year" }); t.textContent = String(yr); svg.appendChild(t);
    }
    const t22 = svgEl("text", { x: X0 + 4, y: TOP - 16, class: "lg-year" }); t22.textContent = "2022.09"; svg.appendChild(t22);

    // edges
    const eg = svgEl("g", { class: "lg-edges" });
    graph.edges = [];
    EDGES.forEach(([a, b]) => {
      const A = graph.nodes[a], B = graph.nodes[b]; if (!A || !B) return;
      const x1 = A.x + A.w, y1 = A.y + 11, x2 = B.x, y2 = B.y + 11;
      const dx = Math.max(30, Math.abs(x2 - x1) * 0.45);
      const path = svgEl("path", { d: `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2 - 2},${y2}`, class: "lg-edge", "marker-end": "url(#lgArrow)" });
      path.dataset.a = a; path.dataset.b = b;
      eg.appendChild(path); graph.edges.push(path);
    });
    svg.appendChild(eg);

    // nodes
    const ng = svgEl("g", { class: "lg-nodes" });
    Object.values(graph.nodes).forEach(n => {
      const g = svgEl("g", { class: "lg-node", tabindex: "0", role: "button", "aria-label": `${n.p.n}，打开解读` });
      g.dataset.k = n.p.k;
      g.appendChild(svgEl("rect", { x: n.x, y: n.y, width: n.w, height: 22, rx: 5 }));
      const t = svgEl("text", { x: n.x + 10, y: n.y + 15 }); t.textContent = n.label; g.appendChild(t);
      const title = svgEl("title", {}); title.textContent = `${n.p.n} · ${n.p.d}`; g.appendChild(title);
      n.g = g; ng.appendChild(g);
    });
    svg.appendChild(ng);

    const reach = (k, dir) => {
      const out = new Set([k]); const stack = [k];
      while (stack.length) {
        const c = stack.pop();
        EDGES.forEach(([a, b]) => {
          const nx = dir === "down" ? (a === c ? b : null) : (b === c ? a : null);
          if (nx && !out.has(nx)) { out.add(nx); stack.push(nx); }
        });
      }
      return out;
    };
    const highlight = k => {
      if (!k) { svg.classList.remove("focus"); svg.querySelectorAll(".hl").forEach(e => e.classList.remove("hl", "self")); graph.edges.forEach(e => e.setAttribute("marker-end", "url(#lgArrow)")); return; }
      const up = reach(k, "up"), down = reach(k, "down");
      const set = new Set([...up, ...down]);
      svg.classList.add("focus");
      Object.values(graph.nodes).forEach(n => { n.g.classList.toggle("hl", set.has(n.p.k)); n.g.classList.toggle("self", n.p.k === k); });
      graph.edges.forEach(e => {
        const on = (up.has(e.dataset.a) && up.has(e.dataset.b)) || (down.has(e.dataset.a) && down.has(e.dataset.b));
        e.classList.toggle("hl", on); e.setAttribute("marker-end", on ? "url(#lgArrowHl)" : "url(#lgArrow)");
      });
    };
    ng.addEventListener("mouseover", e => { const g = e.target.closest(".lg-node"); if (g) highlight(g.dataset.k); });
    ng.addEventListener("mouseleave", () => highlight(null));
    ng.addEventListener("focusin", e => { const g = e.target.closest(".lg-node"); if (g) highlight(g.dataset.k); });
    ng.addEventListener("focusout", () => highlight(null));
    ng.addEventListener("click", e => { const g = e.target.closest(".lg-node"); if (g) openPaper(g.dataset.k); });
    ng.addEventListener("keydown", e => {
      const g = e.target.closest(".lg-node");
      if (g && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); openPaper(g.dataset.k); }
    });
  }

  function renderGraphStatus() {
    Object.values(graph.nodes).forEach(n => {
      const s = window.statusOf(n.p.k);
      ["unread", "reading", "understood", "reproduced"].forEach(c => n.g.classList.toggle("st-" + c, c === s));
    });
  }

  /* ---------- refresh ---------- */
  function refresh() {
    renderDashboard();
    renderCardStatus();
    renderGraphStatus();
    refreshDrawer();
    if (typeof applyFilters === "function") applyFilters();
  }

  lsLoad();
  wireDashboard();
  wireDrawerLearning();
  buildGraph();
  refresh();
  setSync("local");
  initCloud();
})();
