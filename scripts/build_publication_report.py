from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "investments_VC.csv"
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"
REPORT = DOCS / "index.html"

STATUS_ORDER = ["operating", "acquired", "closed"]
STATUS_LABELS = {"operating": "Operating", "acquired": "Acquired", "closed": "Closed"}
PALETTE = {
    "ink": "#0b1324",
    "muted": "#5b6475",
    "line": "#d7deea",
    "wash": "#f3f6fb",
    "panel": "#ffffff",
    "operating": "#5578b8",
    "acquired": "#2f9b68",
    "closed": "#d45b64",
    "accent": "#1d4ed8",
    "gold": "#c27a12",
}


def money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:.0f}"


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100 * float(value):.1f}%"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df.drop(columns=[c for c in df.columns if c.startswith("unnamed")], errors="ignore")


def read_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
    df = clean_columns(df).drop_duplicates()
    df = df[df["status"].isin(STATUS_ORDER)].copy()
    df["market"] = df["market"].fillna("Unknown").astype(str).str.strip()
    df.loc[df["market"].eq(""), "market"] = "Unknown"
    for col in [
        "funding_total_usd",
        "funding_rounds",
        "founded_year",
        "seed",
        "venture",
        "equity_crowdfunding",
        "debt_financing",
        "angel",
        "grant",
        "private_equity",
        "round_a",
        "round_b",
        "round_c",
        "round_d",
        "round_e",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["cohort"] = pd.cut(
        df["founded_year"],
        bins=[1900, 1999, 2004, 2008, 2011, 2014],
        labels=["pre-2000", "2000-04", "2005-08", "2009-11", "2012-14"],
        include_lowest=True,
    )
    df["round_bucket"] = pd.cut(
        df["funding_rounds"],
        bins=[0, 1, 3, 6, 100],
        labels=["1 round", "2-3 rounds", "4-6 rounds", "7+ rounds"],
        include_lowest=True,
    )
    return df


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def status_summary(df: pd.DataFrame) -> list[dict]:
    counts = df["status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
    total = int(counts.sum())
    return [
        {"status": status, "label": STATUS_LABELS[status], "count": int(count), "rate": count / total}
        for status, count in counts.items()
    ]


def cohort_summary(df: pd.DataFrame) -> list[dict]:
    table = (
        df.dropna(subset=["cohort"])
        .groupby(["cohort", "status"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STATUS_ORDER, fill_value=0)
    )
    output = []
    for cohort, row in table.iterrows():
        total = int(row.sum())
        item = {"cohort": str(cohort), "n": total}
        for status in STATUS_ORDER:
            item[status] = int(row[status])
            item[f"{status}_rate"] = row[status] / total if total else 0
        output.append(item)
    return output


def formation_summary(df: pd.DataFrame) -> list[dict]:
    counts = (
        df.dropna(subset=["founded_year"])
        .query("founded_year >= 1980 and founded_year <= 2014")
        .groupby("founded_year")
        .size()
        .reset_index(name="companies")
    )
    counts["founded_year"] = counts["founded_year"].astype(int)
    return records(counts)


def funding_summary(df: pd.DataFrame) -> list[dict]:
    summary = (
        df[df["funding_total_usd"] > 0]
        .dropna(subset=["round_bucket"])
        .groupby(["round_bucket", "status"], observed=False)
        .agg(
            median_funding=("funding_total_usd", "median"),
            companies=("name", "size"),
        )
        .reset_index()
    )
    summary["round_bucket"] = summary["round_bucket"].astype(str)
    return records(summary)


def market_scorecard(df: pd.DataFrame, top_n: int = 24) -> pd.DataFrame:
    top_markets = df[df["market"] != "Unknown"]["market"].value_counts().head(top_n).index
    score = (
        df[df["market"].isin(top_markets)]
        .groupby("market")
        .agg(
            n=("name", "size"),
            acquired_rate=("status", lambda x: (x == "acquired").mean()),
            closed_rate=("status", lambda x: (x == "closed").mean()),
            operating_rate=("status", lambda x: (x == "operating").mean()),
            median_funding=("funding_total_usd", "median"),
            median_rounds=("funding_rounds", "median"),
            countries=("country_code", "nunique"),
        )
        .assign(exit_balance=lambda x: x["acquired_rate"] - x["closed_rate"])
        .sort_values(["exit_balance", "n"], ascending=[False, False])
        .reset_index()
    )
    return score


def capital_mix(df: pd.DataFrame) -> list[dict]:
    columns = {
        "seed": "Seed",
        "angel": "Angel",
        "venture": "Venture",
        "round_a": "Round A",
        "round_b": "Round B",
        "round_c": "Round C",
        "round_d": "Round D",
        "round_e": "Round E",
        "private_equity": "Private equity",
        "debt_financing": "Debt",
        "grant": "Grant",
    }
    rows = []
    for col, label in columns.items():
        if col in df.columns:
            total = df[col].fillna(0).clip(lower=0).sum()
            if total > 0:
                rows.append({"type": label, "total": float(total)})
    rows = sorted(rows, key=lambda x: x["total"], reverse=True)
    return rows[:10]


def country_summary(df: pd.DataFrame) -> list[dict]:
    country = (
        df.dropna(subset=["country_code"])
        .groupby("country_code")
        .agg(
            companies=("name", "size"),
            acquired_rate=("status", lambda x: (x == "acquired").mean()),
            closed_rate=("status", lambda x: (x == "closed").mean()),
            median_funding=("funding_total_usd", "median"),
        )
        .sort_values("companies", ascending=False)
        .head(12)
        .reset_index()
    )
    return records(country)


def missingness(df: pd.DataFrame) -> list[dict]:
    columns = ["market", "country_code", "funding_total_usd", "funding_rounds", "founded_year", "city"]
    rows = []
    for col in columns:
        rows.append({"field": col, "missing_rate": float(df[col].isna().mean())})
    return rows


def build_payload(df: pd.DataFrame) -> dict:
    mature = df[df["founded_year"].between(2000, 2008)]
    market_scores = market_scorecard(df)
    best_market = market_scores.iloc[0]
    riskiest_market = market_scores.sort_values(["closed_rate", "n"], ascending=[False, False]).iloc[0]
    payload = {
        "generated": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "kpis": {
            "companies": int(len(df)),
            "markets": int(df["market"].nunique()),
            "countries": int(df["country_code"].nunique()),
            "medianFunding": float(df.loc[df["funding_total_usd"] > 0, "funding_total_usd"].median()),
            "matureAcquiredRate": float((mature["status"] == "acquired").mean()),
            "matureClosedRate": float((mature["status"] == "closed").mean()),
        },
        "status": status_summary(df),
        "cohorts": cohort_summary(df),
        "formation": formation_summary(df),
        "funding": funding_summary(df),
        "markets": records(market_scores),
        "capitalMix": capital_mix(df),
        "countries": country_summary(df),
        "missingness": missingness(df),
        "narrative": {
            "bestMarket": best_market["market"],
            "bestBalance": float(best_market["exit_balance"]),
            "riskiestMarket": riskiest_market["market"],
            "riskiestClosedRate": float(riskiest_market["closed_rate"]),
        },
    }
    return payload


def text(x: float, y: float, value: str, size: int = 12, weight: int = 400, color: str = PALETTE["ink"], anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">'
        f"{escape(value)}</text>"
    )


def svg_frame(width: int, height: int, body: str) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">{body}</svg>'


def save_static_svg_exports(payload: dict) -> None:
    # Static exports are kept for README previews. The interactive report renders its own
    # responsive SVGs with tooltips, so these snapshots prioritize safe spacing.
    FIGURES.mkdir(parents=True, exist_ok=True)
    status = payload["status"]
    width, height = 980, 360
    parts = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(34, 46, "Current Status Snapshot", 24, 750),
        text(34, 74, "Operating status dominates, so outcome rates require cohort context.", 14, 400, PALETTE["muted"]),
    ]
    left, top, chart_w, bar_h = 220, 120, 650, 44
    max_count = max(d["count"] for d in status)
    for i, row in enumerate(status):
        y = top + i * 68
        w = chart_w * row["count"] / max_count
        color = PALETTE[row["status"]]
        parts.append(text(34, y + 29, row["label"], 15, 750))
        parts.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="8" fill="{color}"/>')
        parts.append(text(left + w + 14, y + 28, f'{row["count"]:,} ({pct(row["rate"])})', 14, 750))
    (FIGURES / "status_composition.svg").write_text(svg_frame(width, height, "\n".join(parts)), encoding="utf-8")

    markets = payload["markets"][:16]
    width, height = 980, 620
    left, top, chart_w, chart_h = 110, 110, 730, 390
    max_x = max(0.18, max(m["closed_rate"] for m in markets) * 1.12)
    max_y = max(0.18, max(m["acquired_rate"] for m in markets) * 1.12)
    max_n = max(m["n"] for m in markets)
    parts = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(34, 46, "Market Exit Fingerprints", 24, 750),
        text(34, 74, "Bubble size is company count. Better balances appear higher and further left.", 14, 400, PALETTE["muted"]),
        f'<rect x="{left}" y="{top}" width="{chart_w}" height="{chart_h}" fill="{PALETTE["wash"]}" stroke="{PALETTE["line"]}"/>',
    ]
    for tick in np.linspace(0, max_x, 5):
        x = left + chart_w * tick / max_x
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + chart_h}" stroke="{PALETTE["line"]}" />')
        parts.append(text(x, top + chart_h + 30, pct(tick), 12, 400, PALETTE["muted"], "middle"))
    for tick in np.linspace(0, max_y, 5):
        y = top + chart_h - chart_h * tick / max_y
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="{PALETTE["line"]}" />')
        parts.append(text(left - 16, y + 4, pct(tick), 12, 400, PALETTE["muted"], "end"))
    parts.append(text(left + chart_w / 2, height - 42, "Closure rate", 15, 750, anchor="middle"))
    parts.append(f'<text x="28" y="{top + chart_h / 2:.1f}" transform="rotate(-90 28 {top + chart_h / 2:.1f})" font-family="Inter, Arial, sans-serif" font-size="15" font-weight="750" fill="{PALETTE["ink"]}" text-anchor="middle">Acquisition rate</text>')
    for m in markets:
        x = left + chart_w * m["closed_rate"] / max_x
        y = top + chart_h - chart_h * m["acquired_rate"] / max_y
        r = 8 + 23 * np.sqrt(m["n"] / max_n)
        color = PALETTE["acquired"] if m["exit_balance"] >= 0 else PALETTE["closed"]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" fill-opacity="0.72" stroke="white" stroke-width="2"/>')
        if m["market"] in ["Software", "Biotechnology", "Enterprise Software", "Curated Web", "Games", "Advertising"]:
            parts.append(text(x + r + 6, y + 4, m["market"], 12, 750))
    (FIGURES / "market_exit_fingerprint.svg").write_text(svg_frame(width, height, "\n".join(parts)), encoding="utf-8")


def build_report(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    kpis = payload["kpis"]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Venture Outcomes Under Censoring</title>
  <style>
    :root {{
      --ink:#071226; --muted:#586174; --soft:#8790a3; --line:#d9e1ee;
      --wash:#f3f6fb; --panel:#ffffff; --blue:#5578b8; --green:#2f9b68;
      --red:#d45b64; --accent:#1d4ed8; --gold:#c27a12; --shadow:0 18px 50px rgba(14,30,62,.10);
    }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ margin:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:var(--wash); }}
    body::before {{ content:""; position:fixed; inset:0; pointer-events:none; background:radial-gradient(circle at 10% 0%, rgba(29,78,216,.14), transparent 28rem), radial-gradient(circle at 90% 10%, rgba(47,155,104,.11), transparent 28rem); }}
    a {{ color:inherit; }}
    .shell {{ position:relative; max-width:1240px; margin:0 auto; padding:28px 24px 72px; }}
    .topnav {{ position:sticky; top:0; z-index:20; display:flex; align-items:center; justify-content:space-between; gap:16px; padding:12px 0; backdrop-filter:blur(16px); }}
    .navlinks {{ display:flex; flex-wrap:wrap; gap:8px; }}
    .navlinks a {{ text-decoration:none; color:var(--muted); font-size:13px; font-weight:800; padding:8px 12px; border:1px solid rgba(7,18,38,.08); border-radius:999px; background:rgba(255,255,255,.74); }}
    .navlinks a:hover {{ color:var(--accent); border-color:rgba(29,78,216,.25); }}
    .brand {{ font-size:12px; font-weight:900; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); }}
    .hero {{ min-height:86vh; display:grid; align-items:center; padding:48px 0 70px; }}
    .hero-grid {{ display:grid; grid-template-columns:minmax(0,1.1fr) 360px; gap:34px; align-items:end; }}
    .kicker {{ margin:0 0 12px; font-size:12px; font-weight:900; letter-spacing:.16em; text-transform:uppercase; color:var(--accent); }}
    h1 {{ margin:0; font-size:clamp(44px, 7vw, 92px); line-height:.95; letter-spacing:-.055em; max-width:880px; }}
    h2 {{ margin:0; font-size:clamp(28px, 4vw, 48px); line-height:1; letter-spacing:-.035em; }}
    h3 {{ margin:0; font-size:20px; line-height:1.15; letter-spacing:-.02em; }}
    p {{ color:var(--muted); line-height:1.68; }}
    .dek {{ max-width:820px; font-size:19px; }}
    .hero-note {{ border:1px solid var(--line); border-radius:28px; padding:24px; background:rgba(255,255,255,.82); box-shadow:var(--shadow); }}
    .hero-note b {{ display:block; margin-bottom:8px; font-size:18px; }}
    .metric-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin:28px 0 0; }}
    .metric {{ background:var(--panel); border:1px solid var(--line); border-radius:22px; padding:20px; box-shadow:0 10px 28px rgba(14,30,62,.06); }}
    .metric strong {{ display:block; font-size:30px; letter-spacing:-.04em; }}
    .metric span {{ display:block; margin-top:4px; color:var(--muted); font-size:13px; }}
    .section {{ min-height:100vh; display:grid; align-items:center; padding:72px 0; scroll-margin-top:70px; }}
    .section-head {{ display:flex; justify-content:space-between; align-items:end; gap:24px; margin-bottom:20px; }}
    .section-head p {{ max-width:560px; margin:12px 0 0; }}
    .eyebrow {{ margin:0 0 10px; color:var(--accent); font-size:12px; font-weight:900; letter-spacing:.16em; text-transform:uppercase; }}
    .grid-2 {{ display:grid; grid-template-columns:1.1fr .9fr; gap:18px; align-items:stretch; }}
    .grid-3 {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }}
    .card {{ background:rgba(255,255,255,.92); border:1px solid var(--line); border-radius:28px; padding:22px; box-shadow:var(--shadow); overflow:hidden; }}
    .card-title {{ display:flex; align-items:flex-start; justify-content:space-between; gap:16px; margin-bottom:10px; }}
    .subtle {{ color:var(--soft); font-size:13px; margin:4px 0 0; }}
    .chart {{ width:100%; min-height:390px; }}
    .chart.short {{ min-height:300px; }}
    svg {{ display:block; width:100%; height:auto; overflow:visible; }}
    .legend {{ display:flex; flex-wrap:wrap; gap:12px; color:var(--muted); font-size:13px; font-weight:800; }}
    .legend span {{ display:inline-flex; align-items:center; gap:6px; }}
    .dot {{ width:10px; height:10px; border-radius:999px; display:inline-block; }}
    .insight-list {{ display:grid; gap:14px; margin-top:16px; }}
    .insight {{ padding:16px; border:1px solid var(--line); border-radius:18px; background:#fbfcff; }}
    .insight b {{ display:block; color:var(--ink); }}
    .controls {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:14px 0 0; }}
    select, button.toggle {{ border:1px solid var(--line); background:white; color:var(--ink); border-radius:999px; min-height:38px; padding:0 14px; font-weight:800; }}
    button.toggle.active {{ color:white; background:var(--accent); border-color:var(--accent); }}
    .market-panel {{ display:grid; gap:12px; }}
    .market-hero {{ padding:18px; border-radius:20px; background:linear-gradient(135deg,#eef5ff,#f5fff9); border:1px solid var(--line); }}
    .market-hero strong {{ display:block; font-size:28px; letter-spacing:-.03em; }}
    .mini-kpis {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }}
    .mini-kpi {{ border:1px solid var(--line); border-radius:16px; padding:12px; background:white; }}
    .mini-kpi b {{ display:block; font-size:20px; }}
    table {{ width:100%; border-collapse:separate; border-spacing:0; overflow:hidden; border:1px solid var(--line); border-radius:20px; background:white; }}
    th,td {{ padding:12px 14px; border-bottom:1px solid var(--line); text-align:left; font-size:14px; }}
    th {{ color:var(--muted); background:#f7f9fd; font-size:12px; letter-spacing:.08em; text-transform:uppercase; }}
    tr:last-child td {{ border-bottom:0; }}
    .tooltip {{ position:fixed; z-index:50; pointer-events:none; opacity:0; transform:translate(-50%, calc(-100% - 14px)); max-width:260px; padding:10px 12px; border-radius:14px; background:#071226; color:white; box-shadow:0 18px 38px rgba(7,18,38,.24); font-size:13px; line-height:1.45; transition:opacity 120ms ease; }}
    .tooltip b {{ color:white; }}
    .callout {{ border-left:5px solid var(--accent); background:white; border-radius:18px; padding:18px; box-shadow:0 10px 28px rgba(14,30,62,.06); }}
    .footer-note {{ color:var(--soft); font-size:13px; margin-top:20px; }}
    .axis-label {{ fill:var(--muted); font-size:12px; font-weight:800; }}
    .tick {{ fill:var(--muted); font-size:11px; }}
    .gridline {{ stroke:#dfe5ef; stroke-width:1; }}
    .bar-label {{ fill:var(--ink); font-size:12px; font-weight:800; }}
    .mark {{ cursor:pointer; }}
    .mark:hover {{ filter:brightness(1.04); stroke:#071226; stroke-width:2.5; }}
    @media (max-width:900px) {{
      .hero-grid,.grid-2 {{ grid-template-columns:1fr; }}
      .metric-grid,.grid-3 {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      .section {{ min-height:auto; padding:54px 0; }}
    }}
    @media (max-width:620px) {{
      .shell {{ padding:20px 14px 48px; }}
      .metric-grid,.grid-3,.mini-kpis {{ grid-template-columns:1fr; }}
      .topnav {{ align-items:flex-start; flex-direction:column; }}
    }}
  </style>
</head>
<body>
<div class="shell">
  <nav class="topnav">
    <div class="brand">Venture outcomes analysis</div>
    <div class="navlinks">
      <a href="#overview">Overview</a>
      <a href="#cohorts">Cohorts</a>
      <a href="#funding">Funding</a>
      <a href="#markets">Markets</a>
      <a href="#quality">Data quality</a>
    </div>
  </nav>

  <header class="hero" id="overview">
    <div class="hero-grid">
      <div>
        <p class="kicker">Crunchbase startup investment dataset</p>
        <h1>Venture outcomes need time context.</h1>
        <p class="dek">A static startup snapshot can make young companies look healthy simply because they have not had enough time to exit or fail. This report separates descriptive signal from survivorship bias.</p>
        <div class="metric-grid">
          <div class="metric"><strong>{kpis["companies"]:,}</strong><span>companies with known status</span></div>
          <div class="metric"><strong>{kpis["markets"]:,}</strong><span>market labels</span></div>
          <div class="metric"><strong>{kpis["countries"]:,}</strong><span>countries represented</span></div>
          <div class="metric"><strong>{money(kpis["medianFunding"])}</strong><span>median disclosed funding</span></div>
        </div>
      </div>
      <aside class="hero-note">
        <b>Executive readout</b>
        <p>For 2000-2008 founding cohorts, {pct(kpis["matureAcquiredRate"])} are marked acquired and {pct(kpis["matureClosedRate"])} are marked closed. Newer cohorts should not be compared directly without survival-time adjustment.</p>
      </aside>
    </div>
  </header>

  <section class="section" id="cohorts">
    <div>
      <div class="section-head">
        <div>
          <p class="eyebrow">1 / Cohort effect</p>
          <h2>Time at risk changes the story.</h2>
          <p>Operating status dominates the raw data. The more useful question is how outcome mix changes when firms have had enough time to mature.</p>
        </div>
        <div class="legend" aria-label="Status legend">
          <span><i class="dot" style="background:var(--blue)"></i>Operating</span>
          <span><i class="dot" style="background:var(--green)"></i>Acquired</span>
          <span><i class="dot" style="background:var(--red)"></i>Closed</span>
        </div>
      </div>
      <div class="grid-2">
        <article class="card">
          <div class="card-title"><div><h3>Company formation by founding year</h3><p class="subtle">Counts are shown from 1980-2014 to avoid sparse early years and boundary noise.</p></div></div>
          <div id="formationChart" class="chart short"></div>
        </article>
        <article class="card">
          <div class="card-title"><div><h3>Outcome mix by cohort</h3><p class="subtle">Hover each segment for counts and rates.</p></div></div>
          <div id="cohortChart" class="chart short"></div>
        </article>
      </div>
    </div>
  </section>

  <section class="section" id="funding">
    <div>
      <div class="section-head">
        <div>
          <p class="eyebrow">2 / Capital ladder</p>
          <h2>Funding depth is a selection signal.</h2>
          <p>More rounds correspond to larger disclosed funding and different outcome mixes. This is useful signal, but not causal evidence that capital alone produces exits.</p>
        </div>
      </div>
      <div class="grid-2">
        <article class="card">
          <div class="card-title"><div><h3>Median funding by rounds and status</h3><p class="subtle">Log scale. Hover bars for exact medians and sample size.</p></div></div>
          <div id="fundingChart" class="chart"></div>
        </article>
        <article class="card">
          <div class="card-title"><div><h3>Capital type mix</h3><p class="subtle">Top disclosed capital categories by aggregate dollars.</p></div></div>
          <div id="capitalChart" class="chart"></div>
        </article>
      </div>
    </div>
  </section>

  <section class="section" id="markets">
    <div>
      <div class="section-head">
        <div>
          <p class="eyebrow">3 / Market fingerprints</p>
          <h2>Markets separate by exit balance.</h2>
          <p>Click or hover a bubble to inspect a market. Size is company count; higher is more acquired, further right is more closed.</p>
        </div>
        <div class="controls">
          <label for="marketSelect" class="subtle">Spotlight</label>
          <select id="marketSelect"></select>
        </div>
      </div>
      <div class="grid-2">
        <article class="card">
          <div id="marketChart" class="chart"></div>
        </article>
        <aside class="market-panel">
          <div class="market-hero">
            <span class="subtle">Selected market</span>
            <strong id="marketName">-</strong>
            <p id="marketReadout">Click a bubble or choose a market.</p>
          </div>
          <div class="mini-kpis">
            <div class="mini-kpi"><b id="marketCompanies">-</b><span class="subtle">companies</span></div>
            <div class="mini-kpi"><b id="marketFunding">-</b><span class="subtle">median funding</span></div>
            <div class="mini-kpi"><b id="marketAcquired">-</b><span class="subtle">acquired</span></div>
            <div class="mini-kpi"><b id="marketClosed">-</b><span class="subtle">closed</span></div>
          </div>
          <div class="card" style="box-shadow:none">
            <h3>Leaderboard</h3>
            <div class="controls" role="group" aria-label="Leaderboard metric">
              <button class="toggle active" data-metric="exit_balance">Exit balance</button>
              <button class="toggle" data-metric="acquired_rate">Acquisition</button>
              <button class="toggle" data-metric="closed_rate">Closure risk</button>
            </div>
            <div id="leaderboard" style="margin-top:14px"></div>
          </div>
        </aside>
      </div>
    </div>
  </section>

  <section class="section" id="quality">
    <div>
      <div class="section-head">
        <div>
          <p class="eyebrow">4 / Data quality</p>
          <h2>Use the dataset, but do not overclaim.</h2>
          <p>The dataset is valuable for visualization and portfolio analysis, but it is a static, incomplete view of venture-backed companies.</p>
        </div>
      </div>
      <div class="grid-2">
        <article class="card">
          <h3>Geographic concentration</h3>
          <p class="subtle">Top countries by company count.</p>
          <div id="countryChart" class="chart"></div>
        </article>
        <article class="card">
          <h3>Missingness audit</h3>
          <p class="subtle">Fields most relevant to this report.</p>
          <div id="missingChart" class="chart short"></div>
          <div class="insight-list">
            <div class="insight"><b>Recommendation</b><p>For modeling, add transaction-level funding dates and use survival analysis. For presentation, keep this project descriptive and explicit about censoring.</p></div>
          </div>
        </article>
      </div>
      <p class="footer-note">Generated from <code>scripts/build_publication_report.py</code>. Source file: <code>investments_VC.csv</code>. Report generated {payload["generated"]}.</p>
    </div>
  </section>
</div>
<div id="tooltip" class="tooltip"></div>

<script>
const DATA = {data};
const colors = {{ operating:"#5578b8", acquired:"#2f9b68", closed:"#d45b64", accent:"#1d4ed8", red:"#d45b64", gold:"#c27a12" }};
const statusOrder = ["operating", "acquired", "closed"];
const statusLabels = {{ operating:"Operating", acquired:"Acquired", closed:"Closed" }};
let selectedMarket = DATA.markets[0]?.market;
let leaderboardMetric = "exit_balance";
const tooltip = document.getElementById("tooltip");

const fmt = new Intl.NumberFormat("en-US");
function pct(v) {{ return `${{(v * 100).toFixed(1)}}%`; }}
function money(v) {{
  if (v == null || Number.isNaN(v)) return "n/a";
  if (Math.abs(v) >= 1e9) return `$${{(v/1e9).toFixed(1)}}B`;
  if (Math.abs(v) >= 1e6) return `$${{(v/1e6).toFixed(1)}}M`;
  if (Math.abs(v) >= 1e3) return `$${{(v/1e3).toFixed(0)}}K`;
  return `$${{v.toFixed(0)}}`;
}}
function clear(el) {{ el.innerHTML = ""; }}
function svgFor(el, h) {{
  clear(el);
  const w = Math.max(520, el.clientWidth || 760);
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${{w}} ${{h}}`);
  svg.setAttribute("role", "img");
  el.appendChild(svg);
  return {{ svg, w, h }};
}}
function node(tag, attrs = {{}}, text = "") {{
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text) n.textContent = text;
  return n;
}}
function addText(svg, x, y, textValue, attrs = {{}}) {{
  const t = node("text", {{ x, y, ...attrs }}, textValue);
  svg.appendChild(t);
  return t;
}}
function showTip(event, html) {{
  tooltip.innerHTML = html;
  tooltip.style.left = `${{event.clientX}}px`;
  tooltip.style.top = `${{event.clientY}}px`;
  tooltip.style.opacity = "1";
}}
function hideTip() {{ tooltip.style.opacity = "0"; }}

function drawFormation() {{
  const el = document.getElementById("formationChart");
  const {{ svg, w, h }} = svgFor(el, 330);
  const m = {{ l:64, r:24, t:24, b:52 }};
  const xs = DATA.formation.map(d => d.founded_year);
  const ys = DATA.formation.map(d => d.companies);
  const minX = Math.min(...xs), maxX = Math.max(...xs), maxY = Math.max(...ys) * 1.08;
  const x = v => m.l + (v - minX) / (maxX - minX) * (w - m.l - m.r);
  const y = v => h - m.b - v / maxY * (h - m.t - m.b);
  [0, 1500, 3000, 4500, 6000].forEach(t => {{
    const yy = y(t);
    svg.appendChild(node("line", {{ x1:m.l, y1:yy, x2:w-m.r, y2:yy, class:"gridline" }}));
    addText(svg, m.l - 10, yy + 4, fmt.format(t), {{ "text-anchor":"end", class:"tick" }});
  }});
  [1980, 1990, 2000, 2010, 2014].forEach(t => {{
    const xx = x(t);
    svg.appendChild(node("line", {{ x1:xx, y1:m.t, x2:xx, y2:h-m.b, class:"gridline" }}));
    addText(svg, xx, h - 20, String(t), {{ "text-anchor":"middle", class:"tick" }});
  }});
  const d = DATA.formation.map((p, i) => `${{i ? "L" : "M"}}${{x(p.founded_year).toFixed(1)}} ${{y(p.companies).toFixed(1)}}`).join(" ");
  svg.appendChild(node("path", {{ d, fill:"none", stroke:colors.accent, "stroke-width":4, "stroke-linejoin":"round", "stroke-linecap":"round" }}));
  addText(svg, m.l + (w - m.l - m.r)/2, h - 2, "Founding year", {{ "text-anchor":"middle", class:"axis-label" }});
  addText(svg, 14, m.t + 12, "Companies", {{ class:"axis-label" }});
}}

function drawCohorts() {{
  const el = document.getElementById("cohortChart");
  const {{ svg, w, h }} = svgFor(el, 330);
  const m = {{ l:90, r:50, t:30, b:46 }};
  const barH = 34;
  DATA.cohorts.forEach((d, i) => {{
    const y = m.t + i * 48;
    addText(svg, m.l - 12, y + 23, d.cohort, {{ "text-anchor":"end", class:"bar-label" }});
    let x0 = m.l;
    statusOrder.forEach(status => {{
      const rate = d[`${{status}}_rate`];
      const bw = (w - m.l - m.r) * rate;
      const rect = node("rect", {{ x:x0, y, width:bw, height:barH, fill:colors[status], rx: status === "operating" ? 8 : 0, class:"mark" }});
      rect.addEventListener("mousemove", e => showTip(e, `<b>${{d.cohort}} / ${{statusLabels[status]}}</b><br>${{fmt.format(d[status])}} companies<br>${{pct(rate)}} of cohort`));
      rect.addEventListener("mouseleave", hideTip);
      svg.appendChild(rect);
      if (bw > 54) addText(svg, x0 + bw/2, y + 22, pct(rate), {{ "text-anchor":"middle", fill:"white", "font-size":11, "font-weight":800 }});
      x0 += bw;
    }});
    addText(svg, w - m.r + 12, y + 23, `n=${{fmt.format(d.n)}}`, {{ class:"tick" }});
  }});
  [0, .25, .5, .75, 1].forEach(t => addText(svg, m.l + (w-m.l-m.r)*t, h - 18, pct(t), {{ "text-anchor":"middle", class:"tick" }}));
}}

function drawFunding() {{
  const el = document.getElementById("fundingChart");
  const {{ svg, w, h }} = svgFor(el, 430);
  const m = {{ l:92, r:24, t:72, b:78 }};
  const buckets = ["1 round", "2-3 rounds", "4-6 rounds", "7+ rounds"];
  const vals = DATA.funding.map(d => d.median_funding).filter(v => v > 0);
  const lo = 4.4, hi = Math.log10(Math.max(...vals)) + .35;
  const y = v => h - m.b - (Math.log10(v) - lo) / (hi - lo) * (h - m.t - m.b);
  [1e5, 1e6, 1e7, 1e8, 1e9].forEach(t => {{
    const yy = y(t);
    svg.appendChild(node("line", {{ x1:m.l, y1:yy, x2:w-m.r, y2:yy, class:"gridline" }}));
    addText(svg, m.l - 10, yy + 4, money(t), {{ "text-anchor":"end", class:"tick" }});
  }});
  const groupW = (w - m.l - m.r) / buckets.length;
  const barW = Math.min(34, groupW / 5);
  buckets.forEach((bucket, i) => {{
    const cx = m.l + groupW * i + groupW / 2;
    addText(svg, cx, h - 38, bucket, {{ "text-anchor":"middle", class:"bar-label" }});
    statusOrder.forEach((status, j) => {{
      const row = DATA.funding.find(d => d.round_bucket === bucket && d.status === status);
      if (!row || !row.median_funding) return;
      const x = cx + (j - 1) * (barW + 8) - barW / 2;
      const yy = y(row.median_funding);
      const rect = node("rect", {{ x, y:yy, width:barW, height:h-m.b-yy, fill:colors[status], rx:6, class:"mark" }});
      rect.addEventListener("mousemove", e => showTip(e, `<b>${{bucket}} / ${{statusLabels[status]}}</b><br>Median funding: ${{money(row.median_funding)}}<br>Companies: ${{fmt.format(row.companies)}}`));
      rect.addEventListener("mouseleave", hideTip);
      svg.appendChild(rect);
    }});
  }});
  addText(svg, m.l + (w-m.l-m.r)/2, h - 6, "Funding round bucket", {{ "text-anchor":"middle", class:"axis-label" }});
  addText(svg, 14, m.t - 18, "Median disclosed funding (log)", {{ class:"axis-label" }});
}}

function drawHorizontalBars(elId, rows, labelKey, valueKey, formatter, color = colors.accent) {{
  const el = document.getElementById(elId);
  const {{ svg, w, h }} = svgFor(el, 400);
  const m = {{ l:150, r:34, t:24, b:26 }};
  const maxV = Math.max(...rows.map(d => d[valueKey]));
  const barH = Math.min(28, (h - m.t - m.b) / rows.length - 7);
  rows.forEach((d, i) => {{
    const y = m.t + i * ((h - m.t - m.b) / rows.length);
    const bw = (w - m.l - m.r) * d[valueKey] / maxV;
    addText(svg, m.l - 12, y + barH * .7, d[labelKey], {{ "text-anchor":"end", class:"bar-label" }});
    svg.appendChild(node("rect", {{ x:m.l, y, width:bw, height:barH, rx:7, fill:color, opacity:.86 }}));
    addText(svg, m.l + bw + 10, y + barH * .7, formatter(d[valueKey]), {{ class:"tick", "font-weight":800 }});
  }});
}}

function drawCapital() {{
  drawHorizontalBars("capitalChart", DATA.capitalMix, "type", "total", money, colors.gold);
}}

function drawCountries() {{
  drawHorizontalBars("countryChart", DATA.countries, "country_code", "companies", v => fmt.format(v), colors.accent);
}}

function drawMissing() {{
  drawHorizontalBars("missingChart", DATA.missingness, "field", "missing_rate", pct, colors.red);
}}

function drawMarkets() {{
  const el = document.getElementById("marketChart");
  const {{ svg, w, h }} = svgFor(el, 500);
  const m = {{ l:76, r:34, t:24, b:70 }};
  const maxX = Math.max(.18, Math.max(...DATA.markets.map(d => d.closed_rate)) * 1.15);
  const maxY = Math.max(.18, Math.max(...DATA.markets.map(d => d.acquired_rate)) * 1.15);
  const maxN = Math.max(...DATA.markets.map(d => d.n));
  const x = v => m.l + v / maxX * (w - m.l - m.r);
  const y = v => h - m.b - v / maxY * (h - m.t - m.b);
  [0, .25, .5, .75, 1].forEach(t => {{
    const xx = m.l + (w-m.l-m.r)*t;
    const yy = m.t + (h-m.t-m.b)*t;
    svg.appendChild(node("line", {{ x1:xx, y1:m.t, x2:xx, y2:h-m.b, class:"gridline" }}));
    svg.appendChild(node("line", {{ x1:m.l, y1:yy, x2:w-m.r, y2:yy, class:"gridline" }}));
    addText(svg, xx, h - 40, pct(maxX*t), {{ "text-anchor":"middle", class:"tick" }});
    addText(svg, m.l - 12, h - m.b - (h-m.t-m.b)*t + 4, pct(maxY*t), {{ "text-anchor":"end", class:"tick" }});
  }});
  svg.appendChild(node("line", {{ x1:x(.08), y1:m.t, x2:x(.08), y2:h-m.b, stroke:"#b7c2d4", "stroke-dasharray":"4 5" }}));
  svg.appendChild(node("line", {{ x1:m.l, y1:y(.08), x2:w-m.r, y2:y(.08), stroke:"#b7c2d4", "stroke-dasharray":"4 5" }}));
  addText(svg, m.l + (w-m.l-m.r)/2, h - 8, "Closure rate", {{ "text-anchor":"middle", class:"axis-label" }});
  addText(svg, 6, m.t + 12, "Acquisition rate", {{ class:"axis-label" }});
  DATA.markets.forEach(d => {{
    const r = 8 + 24 * Math.sqrt(d.n / maxN);
    const fill = d.exit_balance >= 0 ? colors.green : colors.red;
    const circle = node("circle", {{ cx:x(d.closed_rate), cy:y(d.acquired_rate), r, fill, opacity:d.market === selectedMarket ? .95 : .58, stroke:d.market === selectedMarket ? "#071226" : "white", "stroke-width":d.market === selectedMarket ? 3 : 2, class:"mark" }});
    circle.addEventListener("mousemove", e => showTip(e, `<b>${{d.market}}</b><br>${{fmt.format(d.n)}} companies<br>Acquired: ${{pct(d.acquired_rate)}}<br>Closed: ${{pct(d.closed_rate)}}<br>Median funding: ${{money(d.median_funding)}}`));
    circle.addEventListener("mouseleave", hideTip);
    circle.addEventListener("click", () => {{ selectedMarket = d.market; syncMarketUI(); }});
    svg.appendChild(circle);
    if (d.market === selectedMarket || ["Software","Biotechnology","Enterprise Software","Curated Web","Advertising","Games"].includes(d.market)) {{
      addText(svg, x(d.closed_rate) + r + 5, y(d.acquired_rate) + 4, d.market, {{ class:"tick", "font-weight":800 }});
    }}
  }});
}}

function syncMarketUI() {{
  const selected = DATA.markets.find(d => d.market === selectedMarket) || DATA.markets[0];
  selectedMarket = selected.market;
  document.getElementById("marketSelect").value = selectedMarket;
  document.getElementById("marketName").textContent = selected.market;
  document.getElementById("marketCompanies").textContent = fmt.format(selected.n);
  document.getElementById("marketFunding").textContent = money(selected.median_funding);
  document.getElementById("marketAcquired").textContent = pct(selected.acquired_rate);
  document.getElementById("marketClosed").textContent = pct(selected.closed_rate);
  const balance = selected.exit_balance >= 0 ? "positive" : "negative";
  document.getElementById("marketReadout").textContent = `${{selected.market}} has a ${{balance}} acquisition-minus-closure balance of ${{pct(selected.exit_balance)}} across ${{fmt.format(selected.n)}} companies.`;
  drawMarkets();
  drawLeaderboard();
}}

function drawLeaderboard() {{
  const el = document.getElementById("leaderboard");
  const rows = [...DATA.markets].sort((a,b) => leaderboardMetric === "closed_rate" ? b[leaderboardMetric] - a[leaderboardMetric] : b[leaderboardMetric] - a[leaderboardMetric]).slice(0, 8);
  const maxV = Math.max(...rows.map(d => Math.abs(d[leaderboardMetric])));
  el.innerHTML = rows.map(d => {{
    const width = Math.max(4, Math.abs(d[leaderboardMetric]) / maxV * 100);
    const color = leaderboardMetric === "closed_rate" || d[leaderboardMetric] < 0 ? "var(--red)" : "var(--green)";
    return `<div style="display:grid;grid-template-columns:150px 1fr 58px;gap:10px;align-items:center;margin:9px 0">
      <b style="font-size:13px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${{d.market}}</b>
      <span style="height:10px;background:#edf1f7;border-radius:999px;overflow:hidden"><i style="display:block;width:${{width}}%;height:100%;background:${{color}};border-radius:999px"></i></span>
      <span class="subtle" style="text-align:right">${{pct(d[leaderboardMetric])}}</span>
    </div>`;
  }}).join("");
}}

function initControls() {{
  const select = document.getElementById("marketSelect");
  select.innerHTML = DATA.markets.map(d => `<option value="${{d.market}}">${{d.market}}</option>`).join("");
  select.addEventListener("change", e => {{ selectedMarket = e.target.value; syncMarketUI(); }});
  document.querySelectorAll("button.toggle").forEach(btn => btn.addEventListener("click", () => {{
    document.querySelectorAll("button.toggle").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    leaderboardMetric = btn.dataset.metric;
    drawLeaderboard();
  }}));
}}

function drawAll() {{
  drawFormation();
  drawCohorts();
  drawFunding();
  drawCapital();
  drawCountries();
  drawMissing();
  syncMarketUI();
}}

initControls();
drawAll();
let resizeTimer;
window.addEventListener("resize", () => {{
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(drawAll, 140);
}});
</script>
</body>
</html>
"""


def write_report(payload: dict) -> None:
    REPORT.write_text(build_report(payload), encoding="utf-8")


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    df = read_data()
    payload = build_payload(df)
    save_static_svg_exports(payload)
    write_report(payload)


if __name__ == "__main__":
    main()
