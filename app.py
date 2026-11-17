import os
import pickle
import warnings

import pandas as pd
from flask import Flask, jsonify, render_template_string, request

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "linear.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

FEATURES = list(model.feature_names_in_)

# Allowed ranges for each input (min, max)
LIMITS = {
    "Square_Footage": (300, 12000),
    "Num_Bedrooms": (1, 10),
    "Num_Bathrooms": (1, 8),
    "Year_Built": (1900, 2026),
    "Lot_Size": (0.05, 20),
    "Garage_Size": (0, 6),
    "Neighborhood_Quality": (1, 10),
}

app = Flask(__name__)

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HomeWorth · House Price Estimator</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,800&family=Manrope:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --ink: #0b1d2e;
    --deep: #112b42;
    --panel: rgba(255, 255, 255, 0.07);
    --line: rgba(255, 255, 255, 0.14);
    --text: #eaf2f8;
    --muted: #9bb3c6;
    --mint: #3ff0c0;
    --sun: #ffb84d;
    --coral: #ff6b5b;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html { color-scheme: dark; }
  body {
    min-height: 100vh;
    font-family: "Manrope", system-ui, sans-serif;
    color: var(--text);
    background: linear-gradient(160deg, var(--ink), var(--deep) 55%, #1a2f55);
    background-attachment: fixed;
    overflow-x: hidden;
  }

  /* drifting colour glows */
  .glow { position: fixed; border-radius: 50%; filter: blur(90px); opacity: .35; z-index: 0; pointer-events: none; }
  .glow.a { width: 420px; height: 420px; background: var(--mint); top: -120px; left: -100px; animation: drift 18s ease-in-out infinite alternate; }
  .glow.b { width: 380px; height: 380px; background: var(--coral); bottom: -140px; right: -90px; animation: drift 22s ease-in-out infinite alternate-reverse; }
  .glow.c { width: 300px; height: 300px; background: var(--sun); top: 45%; left: 55%; opacity: .16; animation: drift 26s ease-in-out infinite alternate; }
  @keyframes drift { to { transform: translate(70px, 50px) scale(1.15); } }

  .wrap { position: relative; z-index: 1; max-width: 1080px; margin: 0 auto; padding: 48px 20px 90px; }

  header { text-align: center; margin-bottom: 40px; }
  h1 {
    font-family: "Fraunces", Georgia, serif;
    font-weight: 800;
    font-size: clamp(2.4rem, 6vw, 4rem);
    letter-spacing: -0.02em;
    line-height: 1.05;
  }
  h1 span {
    background: linear-gradient(90deg, var(--mint), var(--sun), var(--coral), var(--mint));
    background-size: 300% 100%;
    -webkit-background-clip: text; background-clip: text;
    color: transparent;
    animation: shimmer 8s linear infinite;
  }
  @keyframes shimmer { to { background-position: 300% 0; } }
  header p { color: var(--muted); margin-top: 14px; font-size: 1.05rem; max-width: 520px; margin-inline: auto; }

  /* skyline */
  .skyline { display: block; width: 100%; max-width: 760px; margin: 28px auto 0; height: auto; }
  .win { fill: var(--sun); opacity: .25; animation: twinkle 3.6s ease-in-out infinite; }
  @keyframes twinkle { 50% { opacity: .95; } }

  .grid { display: grid; grid-template-columns: 1.25fr 1fr; gap: 24px; align-items: start; }
  @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }

  .card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 28px;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
  }
  .card h2 { font-family: "Fraunces", serif; font-size: 1.35rem; margin-bottom: 20px; }

  .fields { display: grid; grid-template-columns: 1fr 1fr; gap: 20px 22px; }
  @media (max-width: 520px) { .fields { grid-template-columns: 1fr; } }
  .field.full { grid-column: 1 / -1; }
  .field label { display: flex; justify-content: space-between; font-size: .88rem; color: var(--muted); font-weight: 600; margin-bottom: 8px; }
  .field output { color: var(--mint); font-weight: 700; }

  input[type=range] {
    -webkit-appearance: none; appearance: none; width: 100%; height: 8px; border-radius: 99px;
    background: linear-gradient(90deg, var(--mint) var(--fill, 50%), rgba(255,255,255,.14) var(--fill, 50%));
    outline: none; cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none; width: 22px; height: 22px; border-radius: 50%;
    background: #fff; border: 4px solid var(--mint); transition: transform .2s;
  }
  input[type=range]::-moz-range-thumb { width: 14px; height: 14px; border-radius: 50%; background: #fff; border: 4px solid var(--mint); }
  input[type=range]:hover::-webkit-slider-thumb { transform: scale(1.18); }
  input[type=range]:focus-visible { outline: 2px solid var(--sun); outline-offset: 5px; }

  button {
    margin-top: 28px; width: 100%; padding: 16px; border: 0; border-radius: 14px; cursor: pointer;
    font: 700 1.05rem "Manrope", sans-serif; color: #062018;
    background: linear-gradient(90deg, var(--mint), var(--sun));
    transition: transform .2s, box-shadow .2s;
    box-shadow: 0 10px 30px rgba(63, 240, 192, .25);
  }
  button:hover { transform: translateY(-2px); box-shadow: 0 14px 36px rgba(63, 240, 192, .4); }
  button:active { transform: translateY(0) scale(.99); }
  button:focus-visible { outline: 3px solid #fff; outline-offset: 3px; }
  button[disabled] { opacity: .7; cursor: wait; }

  /* result */
  .result { text-align: center; position: sticky; top: 24px; }
  .house { width: 120px; height: 100px; margin: 0 auto 6px; }
  .house .roof { fill: var(--coral); }
  .house .body { fill: var(--sun); }
  .house .door { fill: var(--deep); }
  .house .glass { fill: var(--mint); }
  .house.bob { animation: bob 2.6s ease-in-out infinite; }
  @keyframes bob { 50% { transform: translateY(-8px); } }
  .label { color: var(--muted); font-size: .9rem; }
  .price {
    font-family: "Fraunces", serif; font-weight: 800;
    font-size: clamp(2.4rem, 5.5vw, 3.4rem); margin: 8px 0 4px;
    background: linear-gradient(90deg, var(--mint), var(--sun));
    -webkit-background-clip: text; background-clip: text; color: transparent;
    font-variant-numeric: tabular-nums;
  }
  .range { color: var(--muted); font-size: .9rem; min-height: 1.3em; }
  .error { color: var(--coral); font-weight: 600; margin-top: 12px; min-height: 1.3em; }
  .pop { animation: pop .6s cubic-bezier(.2, 1.6, .4, 1); }
  @keyframes pop { from { transform: scale(.85); opacity: .3; } }

  footer { text-align: center; color: var(--muted); font-size: .8rem; margin-top: 50px; }

  .entrance { animation: rise .9s cubic-bezier(.2, .8, .2, 1) both; }
  .entrance.d2 { animation-delay: .15s; }
  @keyframes rise { from { opacity: 0; transform: translateY(28px); } }

  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition: none !important; }
  }
</style>
</head>
<body>
<div class="glow a"></div><div class="glow b"></div><div class="glow c"></div>

<div class="wrap">
  <header class="entrance">
    <h1>What is your home <span>worth?</span></h1>
    <p>Move the sliders to describe the property and get an instant price estimate from a linear regression model.</p>

    <svg class="skyline" viewBox="0 0 760 130" aria-hidden="true">
      <g fill="#0f2a40" stroke="rgba(255,255,255,.12)">
        <rect x="20" y="60" width="70" height="70"/><rect x="100" y="30" width="60" height="100"/>
        <rect x="170" y="75" width="80" height="55"/><rect x="260" y="15" width="70" height="115"/>
        <rect x="340" y="55" width="90" height="75"/><rect x="440" y="35" width="60" height="95"/>
        <rect x="510" y="70" width="80" height="60"/><rect x="600" y="25" width="70" height="105"/>
        <rect x="680" y="65" width="60" height="65"/>
      </g>
      <g id="windows"></g>
    </svg>
  </header>

  <div class="grid">
    <section class="card entrance d2">
      <h2>Property details</h2>
      <form id="form" class="fields" autocomplete="off">
        <div class="field full">
          <label for="Square_Footage">Living area (sq ft) <output></output></label>
          <input type="range" id="Square_Footage" min="500" max="6000" step="50" value="1800">
        </div>
        <div class="field">
          <label for="Num_Bedrooms">Bedrooms <output></output></label>
          <input type="range" id="Num_Bedrooms" min="1" max="8" step="1" value="3">
        </div>
        <div class="field">
          <label for="Num_Bathrooms">Bathrooms <output></output></label>
          <input type="range" id="Num_Bathrooms" min="1" max="6" step="1" value="2">
        </div>
        <div class="field">
          <label for="Year_Built">Year built <output></output></label>
          <input type="range" id="Year_Built" min="1900" max="2026" step="1" value="2000">
        </div>
        <div class="field">
          <label for="Lot_Size">Lot size <output></output></label>
          <input type="range" id="Lot_Size" min="0.1" max="5" step="0.1" value="0.5">
        </div>
        <div class="field">
          <label for="Garage_Size">Garage spaces <output></output></label>
          <input type="range" id="Garage_Size" min="0" max="4" step="1" value="2">
        </div>
        <div class="field">
          <label for="Neighborhood_Quality">Neighborhood quality (1–10) <output></output></label>
          <input type="range" id="Neighborhood_Quality" min="1" max="10" step="1" value="6">
        </div>
        <div class="field full">
          <button type="submit" id="btn">Estimate price</button>
        </div>
      </form>
    </section>

    <aside class="card result entrance d2">
      <svg class="house" id="house" viewBox="0 0 120 100" aria-hidden="true">
        <polygon class="roof" points="60,8 112,48 8,48"/>
        <rect class="body" x="20" y="48" width="80" height="44" rx="3"/>
        <rect class="door" x="52" y="62" width="16" height="30" rx="2"/>
        <rect class="glass" x="28" y="58" width="16" height="14" rx="2"/>
        <rect class="glass" x="76" y="58" width="16" height="14" rx="2"/>
      </svg>
      <div class="label">Estimated market value</div>
      <div class="price" id="price">$ —</div>
      <div class="range" id="range">Adjust the sliders and press estimate.</div>
      <div class="error" id="error" role="alert"></div>
    </aside>
  </div>

  <footer>Estimates are for illustration only and are not a formal valuation.</footer>
</div>

<script>
  const fmt = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
  const sliders = [...document.querySelectorAll("input[type=range]")];

  function paint(el) {
    const pct = ((el.value - el.min) / (el.max - el.min)) * 100;
    el.style.setProperty("--fill", pct + "%");
    const out = el.parentElement.querySelector("output");
    out.textContent = el.id === "Square_Footage" ? Number(el.value).toLocaleString() :
                      el.id === "Lot_Size" ? Number(el.value).toFixed(1) : el.value;
  }
  sliders.forEach(s => { paint(s); s.addEventListener("input", () => paint(s)); });

  // twinkling skyline windows
  const g = document.getElementById("windows");
  const ns = "http://www.w3.org/2000/svg";
  [[20,60,70],[100,30,60],[170,75,80],[260,15,70],[340,55,90],[440,35,60],[510,70,80],[600,25,70],[680,65,60]]
    .forEach(([x, y, w]) => {
      for (let wy = y + 12; wy < 118; wy += 18) {
        for (let wx = x + 10; wx < x + w - 10; wx += 16) {
          const r = document.createElementNS(ns, "rect");
          r.setAttribute("x", wx); r.setAttribute("y", wy);
          r.setAttribute("width", 7); r.setAttribute("height", 9);
          r.setAttribute("class", "win");
          r.style.animationDelay = (Math.random() * 3.6).toFixed(2) + "s";
          r.style.animationDuration = (2.4 + Math.random() * 3).toFixed(2) + "s";
          g.appendChild(r);
        }
      }
    });

  // count-up animation
  let current = 0;
  function countTo(target) {
    const el = document.getElementById("price");
    const start = current, t0 = performance.now(), dur = 1100;
    function tick(now) {
      const p = Math.min((now - t0) / dur, 1);
      const eased = 1 - Math.pow(1 - p, 4);
      el.textContent = fmt.format(start + (target - start) * eased);
      if (p < 1) requestAnimationFrame(tick); else current = target;
    }
    requestAnimationFrame(tick);
  }

  document.getElementById("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const btn = document.getElementById("btn");
    const err = document.getElementById("error");
    const rng = document.getElementById("range");
    const house = document.getElementById("house");
    err.textContent = "";
    btn.disabled = true; btn.textContent = "Calculating…";

    const payload = {};
    sliders.forEach(s => payload[s.id] = parseFloat(s.value));

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Something went wrong.");
      countTo(data.prediction);
      rng.textContent = "Likely range " + fmt.format(data.prediction * 0.92) + " – " + fmt.format(data.prediction * 1.08);
      house.classList.add("bob");
      const card = document.querySelector(".result");
      card.classList.remove("pop"); void card.offsetWidth; card.classList.add("pop");
    } catch (ex) {
      err.textContent = ex.message;
    } finally {
      btn.disabled = false; btn.textContent = "Estimate price";
    }
  });
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    values = []
    for name in FEATURES:
        if name not in data:
            return jsonify(error=f"Missing value: {name.replace('_', ' ')}"), 400
        try:
            v = float(data[name])
        except (TypeError, ValueError):
            return jsonify(error=f"{name.replace('_', ' ')} must be a number."), 400
        lo, hi = LIMITS.get(name, (float("-inf"), float("inf")))
        if not lo <= v <= hi:
            return jsonify(error=f"{name.replace('_', ' ')} must be between {lo} and {hi}."), 400
        values.append(v)

    X = pd.DataFrame([values], columns=FEATURES)
    pred = float(model.predict(X)[0])
    return jsonify(prediction=max(pred, 0))


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
