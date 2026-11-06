// On-track replay: cars move along the smoothed track, interpolated between lap times.
// Runs inside a gr.HTML component: `element`, `props` and `watch` are provided by Gradio.
// props.value is a JSON string made by f1agent.planner.replay_data().

const SPEEDS = [0.5, 1, 2, 4, 8];
const SECONDS_PER_LAP_1X = 2.0; // at 1x, one race lap takes 2 s of screen time

function setup() {
  if (element._rpStop) element._rpStop();
  const box = element.querySelector(".rp");
  if (!box) return;
  let data = null;
  try { data = props.value ? JSON.parse(props.value) : null; } catch (e) { data = null; }
  const empty = box.querySelector(".rp-empty");
  const body = box.querySelector(".rp-body");
  if (!data) { empty.style.display = "flex"; body.style.display = "none"; return; }
  empty.style.display = "none"; body.style.display = "flex";

  const canvas = box.querySelector("canvas");
  const ctx = canvas.getContext("2d");
  const board = box.querySelector(".rp-board");
  const playBtn = box.querySelector(".rp-play");
  const speedBtn = box.querySelector(".rp-speed");
  const scrub = box.querySelector(".rp-scrub");
  const lapLabel = box.querySelector(".rp-lap");
  const tip = box.querySelector(".rp-tip");
  box.querySelector(".rp-title").textContent = data.title;

  const tr = data.track, T = data.lap_time, L = data.laps;
  const F = tr.f, X = tr.x, Y = tr.y, N = F.length;
  const cars = data.cars;
  const end = Math.max(...cars.map(c => Math.max(...c.cum.filter(v => v !== null))));
  let tau = 0, playing = false, speedIx = 1, last = null, raf = null, screen = [], lastBoard = 0;
  scrub.max = end.toFixed(1);

  // ── geometry ──
  let W = 0, H = 0, scale = 1, ox = 0, oy = 0;
  function resize() {
    const r = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    W = Math.max(300, r.width); H = Math.max(260, Math.min(560, W * 0.62));
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = W + "px"; canvas.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const pad = 36;
    const xr = [Math.min(...X), Math.max(...X)], yr = [Math.min(...Y), Math.max(...Y)];
    scale = Math.min((W - 2 * pad) / (xr[1] - xr[0]), (H - 2 * pad) / (yr[1] - yr[0]));
    ox = W / 2 - scale * (xr[0] + xr[1]) / 2; oy = H / 2 + scale * (yr[0] + yr[1]) / 2;
    screen = X.map((x, i) => [ox + scale * x, oy - scale * Y[i]]);
    draw();
  }
  function at(f) { // point on the track at lap fraction f
    f = ((f % 1) + 1) % 1;
    let lo = 0, hi = N - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (F[m] <= f) lo = m; else hi = m; }
    const f0 = F[lo], f1 = hi === lo ? f0 + 1e-9 : F[hi];
    const a = f1 > f0 ? (f - f0) / (f1 - f0) : 0;
    const p = screen[lo], q = screen[hi];
    return [p[0] + a * (q[0] - p[0]), p[1] + a * (q[1] - p[1])];
  }
  function tangent(f) { const a = at(f - 0.002), b = at(f + 0.002); const d = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1; return [(b[0] - a[0]) / d, (b[1] - a[1]) / d]; }

  // ── race state at time tau ──
  function state() {
    const out = [];
    for (const c of cars) {
      const cum = c.cum;
      let i = 0;
      while (i + 1 < cum.length && cum[i + 1] !== null && cum[i + 1] <= tau) i++;
      let progress, out_ = false, done = false;
      if (i + 1 >= cum.length) { progress = L; done = true; }
      else if (cum[i + 1] === null) { progress = i; out_ = true; }
      else if (tau < cum[0]) { progress = (tau - cum[0]) / T; }
      else { progress = i + (tau - cum[i]) / (cum[i + 1] - cum[i]); }
      out.push({ c, progress, out: out_, done });
    }
    const running = out.filter(s => !s.out).sort((a, b) => b.progress - a.progress);
    const lead = running.length ? running[0].progress : 0;
    running.forEach((s, k) => { s.pos = k + 1; s.gap = (lead - s.progress) * T; });
    const outs = out.filter(s => s.out);
    return { order: running.concat(outs), lead };
  }

  // ── drawing ──
  function draw() {
    ctx.clearRect(0, 0, W, H);
    const path = () => { ctx.beginPath(); screen.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath(); };
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    path(); ctx.strokeStyle = "#05070b"; ctx.lineWidth = 18; ctx.stroke();
    path(); ctx.strokeStyle = "#3a4150"; ctx.lineWidth = 12; ctx.stroke();
    path(); ctx.strokeStyle = "rgba(255,255,255,0.06)"; ctx.lineWidth = 1; ctx.setLineDash([6, 8]); ctx.stroke(); ctx.setLineDash([]);
    // start/finish: chequered bar across the track
    const s = at(0), t = tangent(0), nx = -t[1], ny = t[0];
    for (let k = -3; k < 3; k++) for (let j = 0; j < 2; j++) {
      ctx.fillStyle = (k + j) % 2 ? "#f5f5f5" : "#111";
      const cx = s[0] + nx * (k + 0.5) * 3 + t[0] * (j - 0.5) * 3, cy = s[1] + ny * (k + 0.5) * 3 + t[1] * (j - 0.5) * 3;
      ctx.fillRect(cx - 1.5, cy - 1.5, 3, 3);
    }
    // direction arrow just after the line
    const a = at(0.025), ta = tangent(0.025);
    ctx.fillStyle = "#9aa3b2"; ctx.beginPath();
    ctx.moveTo(a[0] + ta[0] * 7, a[1] + ta[1] * 7);
    ctx.lineTo(a[0] - ta[0] * 4 - ta[1] * 5, a[1] - ta[1] * 4 + ta[0] * 5);
    ctx.lineTo(a[0] - ta[0] * 4 + ta[1] * 5, a[1] - ta[1] * 4 - ta[0] * 5); ctx.fill();
    // sector boundaries
    ctx.font = "600 11px 'Titillium Web', system-ui, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    [["S1", 0], ["S2", tr.sectors[0]], ["S3", tr.sectors[1]]].forEach(([name, f]) => {
      if (f === null || f === undefined) return;
      const p = at(f), tg = tangent(f);
      ctx.fillStyle = "#9aa3b2"; ctx.fillText(name, p[0] - tg[1] * 20, p[1] + tg[0] * 20);
    });
    // cars, back to front so leaders are drawn on top
    const st = state();
    const drawn = [];
    st.order.slice().reverse().forEach(s => {
      if (s.out) return;
      const p = at(s.progress);
      const focus = s.c.name === data.focus;
      const r = focus ? 8.5 : 6.5;
      ctx.beginPath(); ctx.arc(p[0], p[1], r + 2, 0, 2 * Math.PI); ctx.fillStyle = focus ? "#ffffff" : "#05070b"; ctx.fill();
      ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 2 * Math.PI); ctx.fillStyle = s.c.colour; ctx.fill();
      drawn.push({ s, x: p[0], y: p[1] });
      if (focus || s.pos === 1) {
        const label = (s.pos ? "P" + s.pos + " " : "") + s.c.name;
        ctx.font = "700 12px 'Titillium Web', system-ui, sans-serif";
        const w = ctx.measureText(label).width + 10;
        ctx.fillStyle = "rgba(5,7,11,0.85)"; ctx.fillRect(p[0] + 10, p[1] - 20, w, 18);
        ctx.fillStyle = focus ? "#ffffff" : "#d6dbe4"; ctx.textAlign = "left"; ctx.fillText(label, p[0] + 15, p[1] - 11);
        ctx.textAlign = "center";
      }
    });
    canvas._drawn = drawn;
    const lap = Math.min(L, Math.max(1, Math.floor(st.lead) + 1));
    lapLabel.textContent = (st.lead >= L ? "Finish" : "Lap " + lap) + " / " + L;
    if (!scrub._dragging) scrub.value = tau.toFixed(1);
    const now = performance.now();
    if (now - lastBoard > 120) { lastBoard = now; renderBoard(st); }
  }

  function renderBoard(st) {
    board.innerHTML = st.order.map(s => {
      const focus = s.c.name === data.focus ? " rp-focus" : "";
      const gap = s.out ? "OUT" : s.pos === 1 ? "Leader" : s.gap >= T ? "+" + Math.floor(s.gap / T) + " lap" : "+" + s.gap.toFixed(1);
      return `<div class="rp-row${focus}"><span class="rp-pos">${s.out ? "–" : s.pos}</span>` +
        `<span class="rp-chip" style="background:${s.c.colour}"></span><span class="rp-name">${s.c.name}</span>` +
        `<span class="rp-gap">${gap}</span></div>`;
    }).join("");
  }

  // ── loop and controls ──
  function frame(ts) {
    if (last === null) last = ts;
    const dt = (ts - last) / 1000; last = ts;
    if (playing) {
      tau += dt * SPEEDS[speedIx] * T / SECONDS_PER_LAP_1X;
      if (tau >= end) { tau = end; playing = false; playBtn.textContent = "▶ Play"; }
    }
    draw();
    raf = requestAnimationFrame(frame);
  }
  playBtn.onclick = () => {
    if (tau >= end) tau = 0;
    playing = !playing; playBtn.textContent = playing ? "❚❚ Pause" : "▶ Play";
  };
  speedBtn.onclick = () => { speedIx = (speedIx + 1) % SPEEDS.length; speedBtn.textContent = SPEEDS[speedIx] + "×"; };
  scrub.oninput = () => { scrub._dragging = true; tau = parseFloat(scrub.value); };
  scrub.onchange = () => { scrub._dragging = false; };
  canvas.onmousemove = (e) => {
    const r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = null, bd = 14;
    (canvas._drawn || []).forEach(d => { const dd = Math.hypot(d.x - mx, d.y - my); if (dd < bd) { bd = dd; best = d; } });
    if (best) {
      const s = best.s, gap = s.pos === 1 ? "leader" : s.gap >= T ? "+" + Math.floor(s.gap / T) + " lap" : "+" + s.gap.toFixed(1) + " s";
      tip.textContent = `${s.c.name}  P${s.pos}  ${gap}`;
      tip.style.left = (best.x + 12) + "px"; tip.style.top = (best.y + 12) + "px"; tip.style.display = "block";
    } else tip.style.display = "none";
  };
  canvas.onmouseleave = () => { tip.style.display = "none"; };
  speedBtn.textContent = SPEEDS[speedIx] + "×"; playBtn.textContent = "▶ Play";

  const ro = new ResizeObserver(() => resize());
  ro.observe(canvas.parentElement);
  resize();
  raf = requestAnimationFrame(frame);
  element._rpStop = () => { cancelAnimationFrame(raf); ro.disconnect(); };
}

setup();
watch("value", setup);
