// Flow map 精读讲义 · 交互实验
// 所有实验都基于二维高斯混合数据：x ~ Σ_k w_k N(μ_k, S² I)，ε ~ N(0, I)，z_t = (1−t)x + tε（t=1 为噪声，t=0 为数据）。
// 这种数据下后验、边缘速度、条件速度的协方差都有解析式；ODE 流映射用 RK4 数值积分得到。
(function () {
  "use strict";

  /* ---------------- 随机数 ---------------- */
  function rng(seed) {
    let a = seed >>> 0;
    const u = () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    const n = () => { let x = 0, y = 0; while (x === 0) x = u(); while (y === 0) y = u(); return Math.sqrt(-2 * Math.log(x)) * Math.cos(2 * Math.PI * y); };
    return { u, n };
  }

  /* ---------------- 玩具模型 ---------------- */
  function ring(K, R, phase) { const out = []; for (let k = 0; k < K; k++) { const a = (phase || 0) + 2 * Math.PI * k / K; out.push([R * Math.cos(a), R * Math.sin(a)]); } return out; }
  const MODELS = {
    six: { mus: ring(6, 2.2, Math.PI / 6), S: 0.18 },
    circle: { mus: ring(28, 2.0, 0), S: 0.035 }
  };

  function post(M, z, t) {
    const S2 = M.S * M.S, st2 = (1 - t) * (1 - t) * S2 + t * t, K = M.mus.length;
    const lp = new Array(K); let mx = -Infinity;
    for (let k = 0; k < K; k++) {
      const dx = z[0] - (1 - t) * M.mus[k][0], dy = z[1] - (1 - t) * M.mus[k][1];
      lp[k] = -0.5 * (dx * dx + dy * dy) / st2; if (lp[k] > mx) mx = lp[k];
    }
    let s = 0; for (let k = 0; k < K; k++) { lp[k] = Math.exp(lp[k] - mx); s += lp[k]; }
    const g = (1 - t) * S2 / st2;
    let ex = 0, ey = 0, sxx = 0, sxy = 0, syy = 0;
    for (let k = 0; k < K; k++) {
      const p = lp[k] / s;
      const mxk = M.mus[k][0] + g * (z[0] - (1 - t) * M.mus[k][0]);
      const myk = M.mus[k][1] + g * (z[1] - (1 - t) * M.mus[k][1]);
      ex += p * mxk; ey += p * myk; sxx += p * mxk * mxk; sxy += p * mxk * myk; syy += p * myk * myk;
    }
    const c = S2 * t * t / st2;
    return { ex, ey, cxx: c + sxx - ex * ex, cxy: sxy - ex * ey, cyy: c + syy - ey * ey };
  }
  const EX = (M, z, t) => { const q = post(M, z, t); return [q.ex, q.ey]; };
  const V = (M, z, t) => { const q = post(M, z, t); return [(z[0] - q.ex) / t, (z[1] - q.ey) / t]; };

  const TMIN = 1e-3;
  // 真实流映射 X_{t→r}(z)：从 t 积分到 r（r < t），越靠近 r 步长越细
  function flow(M, z, t, r, N) {
    N = N || 80;
    if (r === t) return [z[0], z[1]];
    if (r > t) { // 反向（往噪声方向）积分，只用于小步长
      let x = z[0], y = z[1]; const n = 12, h = (r - t) / n;
      for (let i = 0; i < n; i++) {
        const a = t + i * h, k1 = V(M, [x, y], a), k2 = V(M, [x + h / 2 * k1[0], y + h / 2 * k1[1]], a + h / 2), k3 = V(M, [x + h / 2 * k2[0], y + h / 2 * k2[1]], a + h / 2), k4 = V(M, [x + h * k3[0], y + h * k3[1]], a + h);
        x += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); y += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
      }
      return [x, y];
    }
    const rr = Math.max(r, TMIN);
    let x = z[0], y = z[1];
    let prev = t;
    for (let i = 1; i <= N; i++) {
      const f = 1 - i / N, b = rr + (t - rr) * f * f, a = prev, h = b - a;
      const k1 = V(M, [x, y], a);
      const k2 = V(M, [x + h / 2 * k1[0], y + h / 2 * k1[1]], a + h / 2);
      const k3 = V(M, [x + h / 2 * k2[0], y + h / 2 * k2[1]], a + h / 2);
      const k4 = V(M, [x + h * k3[0], y + h * k3[1]], b);
      x += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
      y += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
      prev = b;
    }
    if (r <= TMIN) return EX(M, [x, y], TMIN);
    return [x, y];
  }
  function path(M, z, t, r, N) {
    N = N || 120;
    const pts = [[z[0], z[1], t]];
    const rr = Math.max(r, TMIN);
    let x = z[0], y = z[1], prev = t;
    for (let i = 1; i <= N; i++) {
      const f = 1 - i / N, b = rr + (t - rr) * f * f, h = b - prev;
      const k1 = V(M, [x, y], prev), k2 = V(M, [x + h / 2 * k1[0], y + h / 2 * k1[1]], prev + h / 2);
      const k3 = V(M, [x + h / 2 * k2[0], y + h / 2 * k2[1]], prev + h / 2), k4 = V(M, [x + h * k3[0], y + h * k3[1]], b);
      x += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); y += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
      prev = b; pts.push([x, y, b]);
    }
    return pts;
  }
  const U = (M, z, t, r, N) => { const X = flow(M, z, t, r, N); return [(z[0] - X[0]) / (t - r), (z[1] - X[1]) / (t - r)]; };
  function jac(M, z, t, r, N) {
    const e = 1e-3, f0 = flow(M, z, t, r, N), fx = flow(M, [z[0] + e, z[1]], t, r, N), fy = flow(M, [z[0], z[1] + e], t, r, N);
    return { a: (fx[0] - f0[0]) / e, b: (fy[0] - f0[0]) / e, c: (fx[1] - f0[1]) / e, d: (fy[1] - f0[1]) / e, f0 };
  }
  function sampleData(M, R, n) { const out = []; for (let i = 0; i < n; i++) { const k = Math.floor(R.u() * M.mus.length); out.push([M.mus[k][0] + M.S * R.n(), M.mus[k][1] + M.S * R.n()]); } return out; }
  function sampleZ(M, R, n, t) { return sampleData(M, R, n).map(x => [(1 - t) * x[0] + t * R.n(), (1 - t) * x[1] + t * R.n()]); }
  const norm = v => Math.hypot(v[0], v[1]);
  const nearestMu = (M, p) => Math.min(...M.mus.map(m => Math.hypot(p[0] - m[0], p[1] - m[1])));

  /* ---------------- 主题与绘图 ---------------- */
  const css = n => getComputedStyle(document.documentElement).getPropertyValue("--" + n).trim();
  function hexRgb(h) { h = h.replace("#", ""); if (h.length === 3) h = h.split("").map(c => c + c).join(""); const n = parseInt(h.slice(0, 6), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  function mix(c1, c2, t) { const a = hexRgb(c1), b = hexRgb(c2); return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`; }
  function alpha(c, a) { const x = hexRgb(c); return `rgba(${x[0]},${x[1]},${x[2]},${a})`; }

  function Plot(canvas, W, H, ext) {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = W * dpr; canvas.height = H * dpr;
    const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const E = ext || 3.6;
    const P = {
      ctx, W, H, E,
      X: x => (x + E) / (2 * E) * W, Y: y => H - (y + E) / (2 * E) * H,
      inv: (px, py) => [px / W * 2 * E - E, (H - py) / H * 2 * E - E],
      clear() { ctx.fillStyle = css("paper"); ctx.fillRect(0, 0, W, H); },
      frame() { ctx.strokeStyle = css("rule"); ctx.lineWidth = 1; ctx.strokeRect(0.5, 0.5, W - 1, H - 1); },
      dot(p, r, fill, stroke) { ctx.beginPath(); ctx.arc(P.X(p[0]), P.Y(p[1]), r, 0, 7); if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); } },
      line(a, b, color, w, dash) { ctx.beginPath(); ctx.setLineDash(dash || []); ctx.moveTo(P.X(a[0]), P.Y(a[1])); ctx.lineTo(P.X(b[0]), P.Y(b[1])); ctx.strokeStyle = color; ctx.lineWidth = w || 1; ctx.stroke(); ctx.setLineDash([]); },
      poly(pts, color, w, dash) { if (pts.length < 2) return; ctx.beginPath(); ctx.setLineDash(dash || []); ctx.moveTo(P.X(pts[0][0]), P.Y(pts[0][1])); for (const p of pts) ctx.lineTo(P.X(p[0]), P.Y(p[1])); ctx.strokeStyle = color; ctx.lineWidth = w || 1; ctx.stroke(); ctx.setLineDash([]); },
      arrow(a, b, color, w) {
        P.line(a, b, color, w || 1.6);
        const ax = P.X(a[0]), ay = P.Y(a[1]), bx = P.X(b[0]), by = P.Y(b[1]), ang = Math.atan2(by - ay, bx - ax), L = 8;
        if (Math.hypot(bx - ax, by - ay) < 3) return;
        ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx - L * Math.cos(ang - 0.4), by - L * Math.sin(ang - 0.4)); ctx.lineTo(bx - L * Math.cos(ang + 0.4), by - L * Math.sin(ang + 0.4)); ctx.closePath(); ctx.fillStyle = color; ctx.fill();
      },
      text(s, x, y, color, align) { ctx.fillStyle = color || css("muted"); ctx.font = "12px Inter, 'PingFang SC', sans-serif"; ctx.textAlign = align || "left"; ctx.fillText(s, x, y); }
    };
    return P;
  }

  // 折线图（可选对数轴）
  function Chart(canvas, W, H, opt) {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = W * dpr; canvas.height = H * dpr;
    const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const L = 56, Rr = 16, T = 14, B = 40;
    const tx = v => opt.logx ? Math.log10(v) : v, ty = v => opt.logy ? Math.log10(v) : v;
    const X = v => L + (tx(v) - tx(opt.x0)) / (tx(opt.x1) - tx(opt.x0)) * (W - L - Rr);
    const Y = v => H - B - (ty(v) - ty(opt.y0)) / (ty(opt.y1) - ty(opt.y0)) * (H - T - B);
    function draw(series) {
      ctx.fillStyle = css("paper"); ctx.fillRect(0, 0, W, H);
      ctx.font = "11px Inter, sans-serif"; ctx.fillStyle = css("muted"); ctx.strokeStyle = css("rule"); ctx.lineWidth = 1;
      (opt.xticks || []).forEach(v => { const x = X(v); ctx.beginPath(); ctx.moveTo(x, T); ctx.lineTo(x, H - B); ctx.stroke(); ctx.textAlign = "center"; ctx.fillText(String(v), x, H - B + 16); });
      (opt.yticks || []).forEach(v => { const y = Y(v); ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - Rr, y); ctx.stroke(); ctx.textAlign = "right"; ctx.fillText(String(v), L - 6, y + 4); });
      ctx.textAlign = "center"; ctx.fillText(opt.xlabel || "", L + (W - L - Rr) / 2, H - 6);
      ctx.save(); ctx.translate(14, T + (H - T - B) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(opt.ylabel || "", 0, 0); ctx.restore();
      series.forEach(s => {
        const col = css(s.color);
        ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(s.dash || []); ctx.beginPath();
        s.pts.forEach((p, i) => { const x = X(p[0]), y = Y(Math.max(p[1], opt.y0)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
        ctx.stroke(); ctx.setLineDash([]);
        s.pts.forEach(p => { ctx.beginPath(); ctx.arc(X(p[0]), Y(Math.max(p[1], opt.y0)), 3, 0, 7); ctx.fillStyle = col; ctx.fill(); });
      });
    }
    return { draw };
  }

  /* ---------------- 小部件 ---------------- */
  let uid = 0;
  function el(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function slider(label, min, max, step, val, fmt) {
    const id = "lab-s-" + (++uid);
    const w = el("label", "lab-ctl"); w.htmlFor = id;
    const name = el("span", "lab-ctl-name", label);
    const inp = el("input"); inp.type = "range"; inp.id = id; inp.min = min; inp.max = max; inp.step = step; inp.value = val;
    const out = el("span", "lab-ctl-val mono");
    const f = fmt || (v => Number(v).toFixed(2));
    const sync = () => { out.textContent = f(+inp.value); };
    inp.addEventListener("input", sync); sync();
    w.append(name, inp, out);
    return { el: w, inp, get v() { return +inp.value; }, set v(x) { inp.value = x; sync(); } };
  }
  function seg(options, val) {
    const w = el("div", "lab-seg"); w.setAttribute("role", "group");
    const btns = options.map(([v, label]) => { const b = el("button", null, label); b.type = "button"; b.dataset.v = v; b.setAttribute("aria-pressed", String(v === val)); w.appendChild(b); return b; });
    let cur = val, cb = () => {};
    w.addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; cur = b.dataset.v; btns.forEach(x => x.setAttribute("aria-pressed", String(x === b))); cb(cur); });
    return { el: w, get v() { return cur; }, on(f) { cb = f; } };
  }
  function legend(items) { const w = el("div", "lab-legend"); items.forEach(([c, t, shape]) => { const s = el("span", "lab-key"); const sw = el("i", "lab-sw " + (shape || "dot")); sw.style.setProperty("--c", `var(--${c})`); s.append(sw, document.createTextNode(t)); w.appendChild(s); }); return w; }
  const fmt = (x, d) => (Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0)) ? x.toExponential(1) : x.toFixed(d == null ? 3 : d);

  function shell(fig, title) {
    const cap = fig.querySelector("figcaption");
    const head = el("div", "lab-head"); head.appendChild(el("span", "lab-label", title));
    const body = el("div", "lab-body"), plots = el("div", "lab-plots"), side = el("div", "lab-side");
    const ctl = el("div", "lab-controls"), read = el("div", "lab-read"); read.setAttribute("aria-live", "polite");
    side.append(ctl, read); body.append(plots, side);
    fig.insertBefore(head, fig.firstChild); fig.insertBefore(body, cap || null);
    return { plots, ctl, read, side };
  }
  function canvas(parent, W, H, label) { const c = el("canvas", "lab-canvas"); c.style.aspectRatio = W + " / " + H; c.setAttribute("role", "img"); c.setAttribute("aria-label", label || "实验图"); parent.appendChild(c); return c; }
  const later = (fn) => setTimeout(fn, 16);

  /* ======================================================
     实验 A：直接回归会塌缩到条件均值
     ====================================================== */
  function labCollapse(fig) {
    const M = MODELS.six, S = shell(fig, "实验 · 直接回归的最优解 vs 真实流映射");
    const P = Plot(canvas(S.plots, 520, 520, "回归最优解与真实流映射终点的散点图"), 520, 520);
    const st = slider("起点时间 t", 0.05, 1, 0.01, 1);
    const show = seg([["both", "两者"], ["reg", "只看回归"], ["flow", "只看流映射"]], "both");
    S.ctl.append(st.el, show.el, legend([["muted", "数据样本"], ["rule", "起点 z_t", "ring"], ["term-b", "回归最优解 E[x | z_t]"], ["term-a", "真实流映射的终点"]]));
    const data = sampleData(M, rng(3), 600);
    function draw() {
      const t = st.v, R = rng(11), zs = sampleZ(M, R, 160, t);
      const reg = zs.map(z => EX(M, z, t)), fl = zs.map(z => flow(M, z, t, 0, 60));
      P.clear(); data.forEach(p => P.dot(p, 1.4, alpha(css("muted"), 0.35)));
      zs.forEach(z => P.dot(z, 2.4, null, css("rule")));
      if (show.v !== "flow") reg.forEach((p, i) => { P.line(zs[i], p, alpha(css("term-b"), 0.18)); P.dot(p, 3, css("term-b")); });
      if (show.v !== "reg") fl.forEach((p, i) => { P.line(zs[i], p, alpha(css("term-a"), 0.18)); P.dot(p, 3, css("term-a")); });
      P.frame();
      const dr = reg.reduce((s, p) => s + nearestMu(M, p), 0) / reg.length, df = fl.reduce((s, p) => s + nearestMu(M, p), 0) / fl.length;
      S.read.innerHTML = `<div class="lab-num"><span>回归最优解离最近数据团中心</span><b class="mono tb">${fmt(dr, 2)}</b></div><div class="lab-num"><span>流映射终点离最近数据团中心</span><b class="mono ta">${fmt(df, 2)}</b></div><p class="lab-note">数据团的半径约为 ${M.S}。t 越大，回归结果越往中间缩；t=1 时所有点都落在数据均值（原点）上。</p>`;
    }
    st.inp.addEventListener("input", draw); show.on(draw); draw();
    return { redraw: draw };
  }

  /* ======================================================
     实验 B：弯曲轨迹与少步 Euler
     ====================================================== */
  function labEuler(fig) {
    const M = MODELS.six, S = shell(fig, "实验 · 弯曲的轨迹与少步 Euler 采样");
    const P = Plot(canvas(S.plots, 520, 520, "ODE 轨迹与 Euler 采样终点"), 520, 520);
    const steps = seg([["1", "1 步"], ["2", "2 步"], ["4", "4 步"], ["8", "8 步"], ["32", "32 步"]], "1");
    S.ctl.append(el("span", "lab-ctl-name", "Euler 步数"), steps.el, legend([["term-a", "真实 ODE 轨迹与终点", "line"], ["term-b", "Euler 采样路径与终点", "line"]]));
    const R = rng(5), zs = Array.from({ length: 26 }, () => [R.n(), R.n()]);
    const paths = zs.map(z => path(M, z, 1, 0, 120));
    const truth = zs.map(z => flow(M, z, 1, 0, 120));
    function euler(z, n) {
      const pts = [[z[0], z[1]]]; let x = z;
      for (let i = 0; i < n; i++) {
        const t = 1 - i / n, r = 1 - (i + 1) / n, v = V(M, x, t);
        x = [x[0] - (t - r) * v[0], x[1] - (t - r) * v[1]]; pts.push(x);
      }
      return pts;
    }
    function draw() {
      const n = +steps.v; P.clear();
      paths.forEach(p => P.poly(p, alpha(css("term-a"), 0.35), 1.2));
      let err = 0;
      zs.forEach((z, i) => {
        const e = euler(z, n); P.poly(e, alpha(css("term-b"), 0.75), 1.2, [4, 3]);
        P.dot(e[e.length - 1], 3.2, css("term-b")); P.dot(truth[i], 3.2, css("term-a")); P.dot(z, 2, null, css("rule"));
        err += Math.hypot(e[e.length - 1][0] - truth[i][0], e[e.length - 1][1] - truth[i][1]);
      });
      P.frame();
      S.read.innerHTML = `<div class="lab-num"><span>Euler 终点与真实终点的平均距离</span><b class="mono tb">${fmt(err / zs.length, 2)}</b></div><p class="lab-note">1 步 Euler 从 t=1 出发，终点正好是 E[x | z_1]，也就是数据均值：和上一个实验里回归的塌缩是同一件事。</p>`;
    }
    steps.on(draw); draw();
    return { redraw: draw };
  }

  /* ======================================================
     实验 C：流映射的三个方程（Eulerian / Lagrangian / 半群）
     ====================================================== */
  function labEquations(fig) {
    const M = MODELS.six, S = shell(fig, "实验 · 流映射满足的三个方程");
    const cv = canvas(S.plots, 520, 520, "一条 ODE 轨迹上的流映射");
    const P = Plot(cv, 520, 520);
    const mode = seg([["euler", "Eulerian：动起点"], ["lag", "Lagrangian：动终点"], ["semi", "半群：拆两步"]], fig.dataset.preset || "euler");
    const which = seg([["true", "真实流映射"], ["bad", "一步 Euler 近似"]], "true");
    const st = slider("起点时间 t", 0.15, 1, 0.01, 0.9), sr = slider("终点时间 r", 0, 0.85, 0.01, 0.15), ss = slider("中间时间 s（半群）", 0, 1, 0.01, 0.5);
    S.ctl.append(mode.el, st.el, sr.el, ss.el, el("span", "lab-ctl-name", "检验对象"), which.el,
      legend([["rule", "ODE 轨迹", "line"], ["accent", "起点 z_t"], ["term-a", "一步跳：从 t 到 r", "line"], ["term-b", "速度 / 两步跳", "line"]]));
    S.read.insertAdjacentHTML("beforebegin", `<p class="lab-hint">点击图中任意位置，换一条从该点（作为 t=1 的噪声）出发的轨迹。</p>`);
    let z1 = [0.45, 0.05], traj = [];
    const onTraj = t => { // 轨迹上时间 t 的点
      for (let i = 1; i < traj.length; i++) if (traj[i][2] <= t) { const a = traj[i - 1], b = traj[i], w = (a[2] - t) / (a[2] - b[2]); return [a[0] + w * (b[0] - a[0]), a[1] + w * (b[1] - a[1])]; }
      return [traj[traj.length - 1][0], traj[traj.length - 1][1]];
    };
    const exact = t => (t >= 0.999 ? [z1[0], z1[1]] : flow(M, z1, 1, t, 160));
    const Xmap = (z, t, r) => which.v === "true" ? flow(M, z, t, r, 120) : (() => { const v = V(M, z, t); return [z[0] - (t - r) * v[0], z[1] - (t - r) * v[1]]; })();
    function rebuild() { traj = path(M, z1, 1, 0, 300); }
    function draw() {
      let t = st.v, r = Math.min(sr.v, t - 0.05); if (r < 0) r = 0;
      let s = Math.min(Math.max(ss.v, r + 0.02), t - 0.02);
      const zt = exact(t), X = Xmap(zt, t, r);
      P.clear(); P.poly(traj, css("rule"), 2);
      P.dot(zt, 5, css("accent"));
      const h = 1e-3;
      let res = 0, title = "";
      if (mode.v === "semi") {
        const Xs = Xmap(zt, t, s), Xsr = Xmap(Xs, s, r);
        P.arrow(zt, Xs, css("term-b"), 1.6); P.arrow(Xs, Xsr, css("term-b"), 1.6); P.dot(Xs, 3.5, css("term-b"));
        P.arrow(zt, X, css("term-a"), 2); P.dot(X, 4.5, css("term-a"));
        res = Math.hypot(X[0] - Xsr[0], X[1] - Xsr[1]); title = "‖X<sub>t→r</sub> − X<sub>s→r</sub>∘X<sub>t→s</sub>‖";
      } else if (mode.v === "lag") {
        P.arrow(zt, X, css("term-a"), 2); P.dot(X, 4.5, css("term-a"));
        const vX = V(M, X, Math.max(r, TMIN)); const sc = 0.18;
        P.arrow(X, [X[0] - sc * vX[0], X[1] - sc * vX[1]], css("term-b"), 1.8);
        const rr = Math.max(r, 2e-3), Xa = Xmap(zt, t, rr), Xb = Xmap(zt, t, rr + h), dX = [(Xb[0] - Xa[0]) / h, (Xb[1] - Xa[1]) / h], vv = V(M, Xa, rr);
        res = Math.hypot(dX[0] - vv[0], dX[1] - vv[1]); title = "‖∂<sub>r</sub>X<sub>t→r</sub> − v(X<sub>t→r</sub>, r)‖";
      } else {
        P.arrow(zt, X, css("term-a"), 2); P.dot(X, 4.5, css("term-a"));
        const v0 = V(M, zt, t); const sc = 0.18;
        P.arrow(zt, [zt[0] - sc * v0[0], zt[1] - sc * v0[1]], css("term-b"), 1.8);
        const zp = exact(t - h), Xp = Xmap(zp, t - h, r);
        res = Math.hypot((X[0] - Xp[0]) / h, (X[1] - Xp[1]) / h); title = "沿轨迹移动起点时终点的变化率 ‖d/dt X<sub>t→r</sub>(z_t)‖";
      }
      P.frame();
      const ok = res < 0.05;
      const desc = {
        euler: "起点沿着轨迹往前或往后滑动，终点应该一动不动。MeanFlow、SoFlow 用的就是这个方程。",
        lag: "终点沿时间移动时，它的移动速度应该等于终点处的速度场（红色箭头）。TVM、Lagrangian 自蒸馏用的是这个方程。",
        semi: "从 t 一步跳到 r，应该等于先跳到 s 再跳到 r。Shortcut、EMF 用的是这个关系。"
      }[mode.v];
      S.read.innerHTML = `<div class="lab-num"><span>${title}</span><b class="mono ${ok ? "ta" : "tb"}">${fmt(res, 4)}</b></div><p class="lab-note">${desc}${which.v === "bad" ? " 换成一步 Euler 近似后残差明显不为零：这些方程能分辨出错误的映射，所以可以当作训练信号。" : " 对真实流映射，残差只剩数值误差。"}</p>`;
    }
    cv.addEventListener("click", e => { const b = cv.getBoundingClientRect(); z1 = P.inv((e.clientX - b.left) / b.width * 520, (e.clientY - b.top) / b.height * 520); rebuild(); draw(); });
    [st, sr, ss].forEach(x => x.inp.addEventListener("input", draw)); mode.on(draw); which.on(draw);
    rebuild(); draw();
    return { redraw: draw };
  }

  /* ======================================================
     实验 D：数值验证 MeanFlow 恒等式
     ====================================================== */
  function labIdentity(fig) {
    const M = MODELS.six, S = shell(fig, "实验 · 数值验证 MeanFlow 恒等式");
    const P = Plot(canvas(S.plots, 520, 520, "平均速度与瞬时速度"), 520, 520);
    const st = slider("t", 0.2, 1, 0.01, 0.85), sr = slider("r", 0, 0.95, 0.01, 0.1);
    S.ctl.append(st.el, sr.el, legend([["term-a", "平均速度 u(z_t, r, t)", "line"], ["term-b", "瞬时速度 v(z_t, t)", "line"], ["rule", "轨迹", "line"]]));
    const z1 = [-0.4, 0.12], traj = path(M, z1, 1, 0, 300);
    function draw() {
      const t = st.v, r = Math.min(sr.v, t - 0.01), h = 1e-3, N = 160;
      const zt = t >= 0.999 ? z1 : flow(M, z1, 1, t, N);
      const u = U(M, zt, t, r, N), v = V(M, zt, t);
      const zp = flow(M, zt, t, t - h, 8), up = U(M, zp, t - h, r, N);
      const dudt = [(u[0] - up[0]) / h, (u[1] - up[1]) / h];
      const rhs = [v[0] - (t - r) * dudt[0], v[1] - (t - r) * dudt[1]];
      const ut = U(M, zt, t - h, r, N), uz = U(M, [zt[0] + h * v[0], zt[1] + h * v[1]], t, r, N);
      const part = [(u[0] - ut[0]) / h + (uz[0] - u[0]) / h, (u[1] - ut[1]) / h + (uz[1] - u[1]) / h];
      P.clear(); P.poly(traj, css("rule"), 2);
      const X = [zt[0] - (t - r) * u[0], zt[1] - (t - r) * u[1]];
      P.dot(X, 3.5, css("term-a")); P.line(zt, X, alpha(css("term-a"), 0.35), 1, [3, 3]);
      const sc = 0.5; P.arrow(zt, [zt[0] - sc * u[0], zt[1] - sc * u[1]], css("term-a"), 2); P.arrow(zt, [zt[0] - sc * v[0], zt[1] - sc * v[1]], css("term-b"), 2);
      P.dot(zt, 5, css("accent")); P.frame();
      const row = (n, a, cls) => `<tr><th>${n}</th><td class="mono ${cls || ""}">${fmt(a[0])}</td><td class="mono ${cls || ""}">${fmt(a[1])}</td></tr>`;
      S.read.innerHTML = `<table class="lab-table"><thead><tr><th></th><th>x 分量</th><th>y 分量</th></tr></thead><tbody>
        ${row("左边 u", u, "ta")}${row("右边 v − (t−r)·du/dt", rhs, "tb")}${row("du/dt（沿轨迹）", dudt)}${row("∂<sub>z</sub>u·v + ∂<sub>t</sub>u", part)}</tbody></table>
        <p class="lab-note">前两行应当相等，这就是式中的恒等式；后两行说明沿轨迹的全导数可以拆成"对输入的方向导数（切向量 v）+ 对 t 的偏导"，正是 JVP 切向量取 (v, 0, 1) 的原因。把 r 拖到接近 t，u 会趋于 v。</p>`;
    }
    [st, sr].forEach(x => x.inp.addEventListener("input", draw)); draw();
    return { redraw: draw };
  }

  /* ======================================================
     实验 E：JVP 切向量带来的噪声（MeanFlow 方差 / iMF）
     ====================================================== */
  function labTangent(fig) {
    const M = MODELS.six, S = shell(fig, "实验 · 切向量噪声被流映射的 Jacobian 放大");
    const P = Plot(canvas(S.plots, 520, 520, "目标噪声的热力图"), 520, 520);
    const st = slider("t", 0.3, 1, 0.01, 0.8), sd = slider("跨度 Δ = t − r", 0.05, 0.8, 0.01, 0.8), sb = slider("切向量混合系数 β", 0, 1, 0.05, +(fig.dataset.beta || 0));
    S.ctl.append(st.el, sd.el, sb.el, legend([["block", "噪声小"], ["term-b", "噪声大"], ["muted", "z_t 样本"]]));
    const G = 46, cells = [], samples = [];
    let busy = false, pending = false;
    function compute(done) {
      if (busy) { pending = true; return; }
      busy = true; S.read.innerHTML = `<p class="lab-note">正在计算 Jacobian…</p>`;
      const t = st.v, r = Math.max(0, t - sd.v), N = 40;
      const zs = sampleZ(M, rng(21), 260, t);
      cells.length = 0; samples.length = 0;
      let i = 0;
      const total = G * G + zs.length;
      (function chunk() {
        const end = Math.min(total, i + 220);
        for (; i < end; i++) {
          const z = i < G * G ? [-3.6 + (i % G + 0.5) * 7.2 / G, -3.6 + (Math.floor(i / G) + 0.5) * 7.2 / G] : zs[i - G * G];
          const J = jac(M, z, t, r, N), q = post(M, z, t), k = 1 / (t * t);
          const rec = { z, J, S: [q.cxx * k, q.cxy * k, q.cyy * k] };
          (i < G * G ? cells : samples).push(rec);
        }
        if (i < total) setTimeout(chunk, 0); else { busy = false; if (pending) { pending = false; compute(done); } else done(); }
      })();
    }
    const trv = (rec, b) => { // Tr(A Σ Aᵀ), A = βI + (1−β)J
      const a = b + (1 - b) * rec.J.a, bb = (1 - b) * rec.J.b, c = (1 - b) * rec.J.c, d = b + (1 - b) * rec.J.d;
      const [sxx, sxy, syy] = rec.S;
      const r1 = [a * sxx + bb * sxy, a * sxy + bb * syy], r2 = [c * sxx + d * sxy, c * sxy + d * syy];
      return (r1[0] * a + r1[1] * bb) + (r2[0] * c + r2[1] * d);
    };
    function draw() {
      if (!cells.length) return;
      const b = sb.v, vals = cells.map(c => trv(c, b)), logs = vals.map(v => Math.log10(Math.max(v, 1e-3)));
      const lo = -1, hi = 3, w = P.W / G;
      P.clear();
      cells.forEach((c, i) => { const f = Math.min(1, Math.max(0, (logs[i] - lo) / (hi - lo))); P.ctx.fillStyle = mix(css("paper"), css("term-b"), Math.pow(f, 0.9)); P.ctx.fillRect((i % G) * w, P.H - (Math.floor(i / G) + 1) * w, w + 0.5, w + 0.5); });
      samples.forEach(s => P.dot(s.z, 1.6, alpha(css("ink"), 0.55)));
      P.frame();
      const sv = samples.map(s => trv(s, b)).sort((x, y) => x - y), mean = sv.reduce((x, y) => x + y, 0) / sv.length;
      const q = p => sv[Math.min(sv.length - 1, Math.floor(p * sv.length))];
      const m0 = samples.reduce((x, s) => x + trv(s, 0), 0) / samples.length, m1 = samples.reduce((x, s) => x + trv(s, 1), 0) / samples.length;
      S.read.innerHTML = `<div class="lab-num"><span>目标噪声 · 均值</span><b class="mono tb">${fmt(mean, 2)}</b></div>
        <div class="lab-num"><span>中位数</span><b class="mono">${fmt(q(0.5), 2)}</b></div>
        <div class="lab-num"><span>99% 分位</span><b class="mono">${fmt(q(0.99), 1)}</b></div>
        <div class="lab-num"><span>最大值</span><b class="mono">${fmt(sv[sv.length - 1], 1)}</b></div>
        <p class="lab-note">对比：β=0（原始 MeanFlow）均值 <b class="mono">${fmt(m0, 2)}</b>；β=1（切向量换成确定的速度）均值 <b class="mono">${fmt(m1, 2)}</b>。色标为 log₁₀，从 0.1 到 1000。</p>`;
    }
    st.inp.addEventListener("change", () => compute(draw)); sd.inp.addEventListener("change", () => compute(draw)); sb.inp.addEventListener("input", draw);
    compute(draw);
    return { redraw: draw };
  }

  /* ======================================================
     实验 E2：目标噪声随跨度 Δ 的变化（TE-MeanFlow）
     ====================================================== */
  function labDelta(fig) {
    const M = MODELS.six, S = shell(fig, "实验 · 不同跨度 Δ 的目标噪声");
    const C = Chart(canvas(S.plots, 560, 340, "目标噪声分位数随跨度变化的折线图"), 560, 340, { x0: 0.05, x1: 0.9, y0: 0.5, y1: 20000, logy: true, xticks: [0.1, 0.3, 0.5, 0.7, 0.9], yticks: [1, 10, 100, 1000, 10000], xlabel: "跨度 Δ = t − r", ylabel: "目标噪声（log）" });
    const tt = seg([["0.9", "t = 0.9"], ["0.6", "t = 0.6"]], "0.9");
    S.ctl.append(el("span", "lab-ctl-name", "起点时间"), tt.el, legend([["term-b", "99% 分位（β=0）", "line"], ["muted", "中位数（β=0）", "line"], ["term-a", "均值（β=1，确定性切向量）", "line"]]));
    let series = null;
    function compute() {
      const t = +tt.v, Ds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].filter(d => d <= t + 1e-9), zs = sampleZ(M, rng(3), 200, t);
      const p99 = [], med = [], m1 = []; let di = 0;
      S.read.innerHTML = `<p class="lab-note">正在计算（约几秒）…</p>`;
      (function step() {
        const D = Ds[di], r = Math.max(0, t - D), vals0 = []; let s1 = 0;
        zs.forEach(z => {
          const J = jac(M, z, t, r, 40), q = post(M, z, t), k = 1 / (t * t), Sg = [q.cxx * k, q.cxy * k, q.cyy * k];
          const tr = (A) => { const [a, b, c, d] = A, [x, y, w] = Sg; return a * (a * x + b * y) + b * (a * y + b * w) + c * (c * x + d * y) + d * (c * y + d * w); };
          vals0.push(tr([J.a, J.b, J.c, J.d])); s1 += tr([1, 0, 0, 1]);
        });
        vals0.sort((a, b) => a - b);
        p99.push([D, vals0[Math.floor(0.99 * vals0.length)]]); med.push([D, vals0[Math.floor(vals0.length / 2)]]); m1.push([D, s1 / zs.length]);
        di++;
        if (di < Ds.length) setTimeout(step, 0); else { series = [{ pts: p99, color: "term-b" }, { pts: med, color: "muted", dash: [4, 3] }, { pts: m1, color: "term-a" }]; draw(); }
      })();
    }
    function draw() {
      if (!series) return; C.draw(series);
      const a = series[0].pts, m = series[1].pts;
      S.read.innerHTML = `<div class="lab-num"><span>99% 分位：最大跨度 / 最小跨度</span><b class="mono tb">${fmt(a[a.length - 1][1] / a[0][1], 0)}×</b></div>
        <div class="lab-num"><span>中位数：最大跨度 / 最小跨度</span><b class="mono">${fmt(m[m.length - 1][1] / m[0][1], 2)}×</b></div>
        <p class="lab-note">跨度越大，流映射在模式分界处的拉伸越强，噪声分布的尾巴越重：99% 分位能增长几个数量级，中位数反而下降。均值被这条长尾主导，在有限样本下会剧烈波动。</p>`;
    }
    tt.on(compute); compute();
    return { redraw: draw };
  }

  /* ======================================================
     实验 F：plug-in 速度估计（ESC）
     ====================================================== */
  function labPlugin(fig) {
    const M = MODELS.six, S = shell(fig, "实验 · 用整批样本估计边缘速度");
    const C = Chart(canvas(S.plots, 560, 340, "两种速度估计的误差随批大小变化"), 560, 340, { x0: 1, x1: 256, y0: 0.01, y1: 10, logx: true, logy: true, xticks: [1, 4, 16, 64, 256], yticks: [0.01, 0.1, 1, 10], xlabel: "批大小 B", ylabel: "与真实边缘速度的均方误差" });
    const tt = seg([["0.2", "t = 0.2"], ["0.5", "t = 0.5"], ["0.9", "t = 0.9"]], "0.5");
    S.ctl.append(el("span", "lab-ctl-name", "时间"), tt.el, legend([["term-b", "条件速度 (z − x)/t", "line"], ["term-a", "plug-in 估计", "line"]]));
    let series = null;
    function compute() {
      const t = +tt.v, Bs = [1, 2, 4, 8, 16, 32, 64, 128, 256], R = rng(41), n = 260, a = [], b = [];
      Bs.forEach(B => {
        let ea = 0, eb = 0;
        for (let i = 0; i < n; i++) {
          const x = sampleData(M, R, 1)[0], z = [(1 - t) * x[0] + t * R.n(), (1 - t) * x[1] + t * R.n()], vt = V(M, z, t);
          const vc = [(z[0] - x[0]) / t, (z[1] - x[1]) / t];
          const xb = [x].concat(sampleData(M, R, B - 1));
          let mx = -Infinity; const lw = xb.map(xj => { const dx = z[0] - (1 - t) * xj[0], dy = z[1] - (1 - t) * xj[1], v = -0.5 * (dx * dx + dy * dy) / (t * t); if (v > mx) mx = v; return v; });
          let sw = 0, px = 0, py = 0; lw.forEach((l, j) => { const w = Math.exp(l - mx); sw += w; px += w * (z[0] - xb[j][0]) / t; py += w * (z[1] - xb[j][1]) / t; });
          ea += (vc[0] - vt[0]) ** 2 + (vc[1] - vt[1]) ** 2; eb += (px / sw - vt[0]) ** 2 + (py / sw - vt[1]) ** 2;
        }
        a.push([B, ea / n]); b.push([B, eb / n]);
      });
      series = [{ pts: a, color: "term-b" }, { pts: b, color: "term-a" }]; draw();
    }
    function draw() {
      if (!series) return; C.draw(series);
      const a = series[0].pts, b = series[1].pts;
      S.read.innerHTML = `<div class="lab-num"><span>B = 256 时误差缩小为</span><b class="mono ta">${fmt(b[b.length - 1][1] / a[a.length - 1][1] * 100, 1)}%</b></div><p class="lab-note">plug-in 估计按 p(z_t | x_j) 给整批样本加权，误差大约按 1/B 下降。高维数据里权重容易集中到少数样本上，收益会小得多，这也是 ESC 只按一定概率使用它、并配合"同类别成批"的原因。</p>`;
    }
    tt.on(() => later(compute)); later(compute);
    return { redraw: draw };
  }

  /* ======================================================
     实验 G：x 场在流形附近，u 场是弥散的（pMF）
     ====================================================== */
  function labManifold(fig) {
    const M = MODELS.circle, S = shell(fig, "实验 · 去噪图像场 x 与平均速度场 u 长什么样");
    const wrap = el("div", "lab-twin"); S.plots.appendChild(wrap);
    const c1 = el("div"), c2 = el("div"); wrap.append(c1, c2);
    c1.appendChild(el("span", "lab-sub", "平均速度 u(z_t, r, t)")); c2.appendChild(el("span", "lab-sub", "去噪图像 x = z_t − t·u"));
    const P1 = Plot(canvas(c1, 300, 300, "u 场的散点"), 300, 300, 4.5), P2 = Plot(canvas(c2, 300, 300, "x 场的散点"), 300, 300, 3);
    const st = slider("t", 0.2, 1, 0.01, 0.9), sr = slider("r", 0, 0.9, 0.01, 0);
    S.ctl.append(st.el, sr.el, legend([["term-b", "u 的取值"], ["term-a", "x 的取值"], ["rule", "数据流形（圆）", "line"]]));
    function draw() {
      const t = st.v, r = Math.min(sr.v, t - 0.02), zs = sampleZ(M, rng(51), 220, t);
      const us = zs.map(z => U(M, z, t, r, 50)), xs = zs.map((z, i) => [z[0] - t * us[i][0], z[1] - t * us[i][1]]);
      [P1, P2].forEach(P => P.clear());
      const circ = []; for (let k = 0; k <= 80; k++) circ.push([2 * Math.cos(k / 80 * 2 * Math.PI), 2 * Math.sin(k / 80 * 2 * Math.PI)]);
      P2.poly(circ, css("rule"), 1.5);
      us.forEach(p => P1.dot(p, 2, alpha(css("term-b"), 0.7))); xs.forEach(p => P2.dot(p, 2, alpha(css("term-a"), 0.75)));
      P1.frame(); P2.frame();
      const dist = xs.reduce((s, p) => s + Math.abs(Math.hypot(p[0], p[1]) - 2), 0) / xs.length;
      const mu = us.reduce((s, p) => [s[0] + p[0] / us.length, s[1] + p[1] / us.length], [0, 0]);
      const spread = Math.sqrt(us.reduce((s, p) => s + (p[0] - mu[0]) ** 2 + (p[1] - mu[1]) ** 2, 0) / us.length);
      S.read.innerHTML = `<div class="lab-num"><span>x 到圆的平均距离</span><b class="mono ta">${fmt(dist, 3)}</b></div><div class="lab-num"><span>u 的散布（标准差）</span><b class="mono tb">${fmt(spread, 2)}</b></div><p class="lab-note">r=0 时 x 就是流映射的终点，落在圆上；r 接近 t 时 x 是去噪估计，会往圆内收。无论哪种，x 都贴着一条一维曲线，而 u 带着噪声铺满二维。高维图像里，这种差别决定了网络能不能拟合。</p>`;
    }
    [st, sr].forEach(x => x.inp.addEventListener("input", draw)); draw();
    return { redraw: draw };
  }

  /* ======================================================
     实验 H：耦合决定直接回归会不会塌缩（OT-NFM）
     ====================================================== */
  function labCoupling(fig) {
    const S = shell(fig, "实验 · 独立耦合 vs 小批量 OT 耦合");
    const C = canvas(S.plots, 560, 360, "噪声到数据的一维映射与输出分布");
    const W = 560, H = 360, dpr = Math.min(2, window.devicePixelRatio || 1); C.width = W * dpr; C.height = H * dpr; const ctx = C.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const sb = slider("小批量大小 B", 0, 10, 1, 6, v => v === 0 ? "独立耦合" : String(2 ** v));
    S.ctl.append(sb.el, legend([["muted", "数据分布"], ["term-a", "学到的映射输出", "line"], ["rule", "配对连线", "line"]]));
    const R = rng(61), n = 2048;
    const eps = Array.from({ length: n }, () => R.n()), xs = Array.from({ length: n }, () => (R.u() < 0.5 ? -2 : 2) + 0.35 * R.n());
    const test = Array.from({ length: 4000 }, () => R.n()), dataT = Array.from({ length: 4000 }, () => (R.u() < 0.5 ? -2 : 2) + 0.35 * R.n()).sort((a, b) => a - b);
    function couple(lv) {
      const idx = xs.map((_, i) => i); if (lv === 0) return idx;
      const B = 2 ** lv, out = new Array(n);
      for (let s = 0; s < n; s += B) {
        const e = [], x = []; for (let i = s; i < Math.min(n, s + B); i++) { e.push(i); x.push(i); }
        e.sort((a, b) => eps[a] - eps[b]); x.sort((a, b) => xs[a] - xs[b]);
        e.forEach((ei, j) => { out[ei] = x[j]; });
      }
      return out;
    }
    function draw() {
      const lv = sb.v, pair = couple(lv);
      const nb = 48, lo = -3.2, hi = 3.2, sum = new Array(nb).fill(0), cnt = new Array(nb).fill(0);
      eps.forEach((e, i) => { const k = Math.min(nb - 1, Math.max(0, Math.floor((e - lo) / (hi - lo) * nb))); sum[k] += xs[pair[i]]; cnt[k]++; });
      const fmap = sum.map((s, k) => cnt[k] ? s / cnt[k] : NaN);
      for (let k = 0; k < nb; k++) if (isNaN(fmap[k])) fmap[k] = k ? fmap[k - 1] : 0;
      const F = e => { const p = (e - lo) / (hi - lo) * nb - 0.5, k = Math.min(nb - 2, Math.max(0, Math.floor(p))), w = Math.min(1, Math.max(0, p - k)); return fmap[k] * (1 - w) + fmap[k + 1] * w; };
      const out = test.map(F).sort((a, b) => a - b);
      ctx.fillStyle = css("paper"); ctx.fillRect(0, 0, W, H);
      const top = 26, lane = 110, xm = v => 40 + (v + 4) / 8 * (W - 80);
      ctx.font = "12px Inter, sans-serif"; ctx.fillStyle = css("muted"); ctx.textAlign = "left";
      ctx.fillText("噪声 ε", 8, top - 8); ctx.fillText("数据 x", 8, top + lane + 22);
      ctx.strokeStyle = alpha(css("rule"), 0.9); ctx.lineWidth = 1;
      for (let i = 0; i < 70; i++) { ctx.beginPath(); ctx.moveTo(xm(eps[i]), top); ctx.lineTo(xm(xs[pair[i]]), top + lane); ctx.stroke(); }
      [top, top + lane].forEach(y => { ctx.strokeStyle = css("rule"); ctx.beginPath(); ctx.moveTo(40, y); ctx.lineTo(W - 40, y); ctx.stroke(); });
      const hy = H - 30, hh = 150, bins = 64, h1 = new Array(bins).fill(0), h2 = new Array(bins).fill(0);
      dataT.forEach(v => { const k = Math.floor((v + 4) / 8 * bins); if (k >= 0 && k < bins) h1[k]++; });
      out.forEach(v => { const k = Math.floor((v + 4) / 8 * bins); if (k >= 0 && k < bins) h2[k]++; });
      const mx = Math.max(...h1, ...h2), bw = (W - 80) / bins;
      h1.forEach((c, k) => { ctx.fillStyle = alpha(css("muted"), 0.35); ctx.fillRect(40 + k * bw, hy - c / mx * hh, bw - 1, c / mx * hh); });
      ctx.strokeStyle = css("term-a"); ctx.lineWidth = 2; ctx.beginPath(); h2.forEach((c, k) => { const x = 40 + (k + 0.5) * bw, y = hy - c / mx * hh; k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
      ctx.strokeStyle = css("rule"); ctx.beginPath(); ctx.moveTo(40, hy); ctx.lineTo(W - 40, hy); ctx.stroke();
      ctx.fillStyle = css("muted"); ctx.textAlign = "center"; [-4, -2, 0, 2, 4].forEach(v => ctx.fillText(String(v), xm(v), hy + 16));
      let w1 = 0; for (let i = 0; i < out.length; i++) w1 += Math.abs(out[i] - dataT[i]); w1 /= out.length;
      S.read.innerHTML = `<div class="lab-num"><span>输出分布与数据分布的 W₁ 距离</span><b class="mono ${w1 < 0.3 ? "ta" : "tb"}">${fmt(w1, 2)}</b></div><p class="lab-note">直接回归"噪声 → 配对的数据"，学到的是条件均值 E[x | ε]。独立耦合下每个 ε 配到的 x 是随机的，均值就是 0，所有输出塌到中间；批越大，排序配对越接近全局 OT（一维时就是按分位数配对），映射越接近确定性的单调映射。</p>`;
    }
    sb.inp.addEventListener("input", draw); draw();
    return { redraw: draw };
  }

  /* ======================================================
     实验 I：有限差分代替 JVP（SoFlow / EMF）
     ====================================================== */
  function labFinite(fig) {
    const M = MODELS.six, S = shell(fig, "实验 · 有限差分步长的取舍");
    const C = Chart(canvas(S.plots, 560, 340, "有限差分误差随步长变化"), 560, 340, { x0: 0.002, x1: 0.3, y0: 1e-4, y1: 10, logx: true, logy: true, xticks: [0.002, 0.01, 0.05, 0.3], yticks: [0.0001, 0.001, 0.01, 0.1, 1, 10], xlabel: "差分步长 h（SoFlow 中的 t − l）", ylabel: "du/dt 的相对误差" });
    const sg = seg([["0.0001", "σ = 10⁻⁴"], ["0.001", "σ = 10⁻³"], ["0.01", "σ = 10⁻²"]], "0.001");
    S.ctl.append(el("span", "lab-ctl-name", "假设网络输出的随机误差 σ"), sg.el, legend([["term-b", "截断误差（真实计算）", "line"], ["muted", "噪声误差 ≈ √2·σ / h", "line"], ["term-a", "两者合计", "line"]]));
    let bias = null, refn = 1;
    function compute() {
      const t = 0.9, r = 0.1, N = 200, zs = sampleZ(M, rng(71), 24, t);
      const hs = [0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.25];
      const out = hs.map(() => 0); let rn = 0;
      zs.forEach(z => {
        const e = 2e-4, a = flow(M, z, t, t - e, 6), b = flow(M, z, t, t + e, 6), ua = U(M, a, t - e, r, N), ub = U(M, b, t + e, r, N);
        const ref = [(ub[0] - ua[0]) / (2 * e), (ub[1] - ua[1]) / (2 * e)], u = U(M, z, t, r, N), nr = Math.max(norm(ref), 1e-6); rn += nr;
        hs.forEach((h, i) => { const zl = flow(M, z, t, t - h, 12), ul = U(M, zl, t - h, r, N); out[i] += Math.hypot((u[0] - ul[0]) / h - ref[0], (u[1] - ul[1]) / h - ref[1]) / nr; });
      });
      bias = hs.map((h, i) => [h, out[i] / zs.length]); refn = rn / zs.length; draw();
    }
    function draw() {
      if (!bias) return;
      const s = +sg.v, noise = bias.map(([h]) => [h, Math.SQRT2 * s / h / refn]), tot = bias.map(([h, b], i) => [h, Math.hypot(b, noise[i][1])]);
      C.draw([{ pts: bias, color: "term-b" }, { pts: noise, color: "muted", dash: [4, 3] }, { pts: tot, color: "term-a" }]);
      let best = tot[0]; tot.forEach(p => { if (p[1] < best[1]) best = p; });
      const slope = (Math.log10(bias[bias.length - 1][1]) - Math.log10(bias[0][1])) / (Math.log10(bias[bias.length - 1][0]) - Math.log10(bias[0][0]));
      S.read.innerHTML = `<div class="lab-num"><span>截断误差的斜率（log-log）</span><b class="mono tb">${fmt(slope, 2)}</b></div><div class="lab-num"><span>合计误差最小的步长</span><b class="mono ta">h ≈ ${fmt(best[0], 3)}</b></div>
        <p class="lab-note">截断误差在二维例子里沿真实轨迹算出，约按 h 线性下降；噪声误差是一个假设模型：若网络输出带 σ 的随机误差，差分会把它放大约 √2·σ/h 倍。两者合计呈 U 形。网络越准（σ 越小），最优步长越小，这正是 SoFlow 随训练逐渐缩小步长的道理。</p>`;
    }
    sg.on(draw); later(compute);
    return { redraw: draw };
  }

  /* ---------------- 挂载 ---------------- */
  const REG = { collapse: labCollapse, euler: labEuler, equations: labEquations, identity: labIdentity, tangent: labTangent, delta: labDelta, plugin: labPlugin, manifold: labManifold, coupling: labCoupling, finite: labFinite };
  let mounted = [];
  window.FlowLab = {
    mountAll(root) {
      mounted = [];
      root.querySelectorAll("figure.lab[data-lab]").forEach(fig => {
        const f = REG[fig.dataset.lab]; if (!f) return;
        try { mounted.push(f(fig)); } catch (e) { fig.insertAdjacentHTML("beforeend", `<p class="lab-note">实验加载失败：${String(e && e.message || e)}</p>`); }
      });
    },
    redrawAll() { mounted.forEach(m => { try { m.redraw(); } catch (e) {} }); },
    _toy: { MODELS, post, flow, V, U, jac }
  };
  const mq = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)");
  if (mq && mq.addEventListener) mq.addEventListener("change", () => setTimeout(() => window.FlowLab.redrawAll(), 30));
  new MutationObserver(() => setTimeout(() => window.FlowLab.redrawAll(), 30)).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
})();
