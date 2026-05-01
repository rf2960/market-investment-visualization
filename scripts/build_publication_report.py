from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "investments_VC.csv"
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"
REPORT = DOCS / "index.html"


PALETTE = {
    "ink": "#172033",
    "muted": "#5d6472",
    "grid": "#d9dee8",
    "operating": "#6b8fbf",
    "acquired": "#4b9b6f",
    "closed": "#c75b62",
    "accent": "#3457d5",
    "gold": "#d89b28",
    "surface": "#f8fafc",
}


def money(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:.0f}"


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def read_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, encoding="ISO-8859-1")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if c.startswith("unnamed")], errors="ignore")
    df = df.drop_duplicates()
    df = df[df["status"].isin(["operating", "acquired", "closed"])].copy()
    df["market"] = df["market"].fillna("Unknown")
    df["funding_total_usd"] = pd.to_numeric(df["funding_total_usd"], errors="coerce")
    df["funding_rounds"] = pd.to_numeric(df["funding_rounds"], errors="coerce")
    df["founded_year"] = pd.to_numeric(df["founded_year"], errors="coerce")
    df["log_funding"] = np.log10(df["funding_total_usd"].clip(lower=1))
    df["round_bucket"] = pd.cut(
        df["funding_rounds"],
        bins=[0, 1, 3, 6, 99],
        labels=["1", "2-3", "4-6", "7+"],
        include_lowest=True,
    )
    df["cohort"] = pd.cut(
        df["founded_year"],
        bins=[1900, 1999, 2004, 2008, 2011, 2014],
        labels=["pre-2000", "2000-04", "2005-08", "2009-11", "2012-14"],
        include_lowest=True,
    )
    return df


def svg_frame(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">{body}</svg>'
    )


def text(x: float, y: float, value: str, size: int = 12, weight: int = 400, color: str = PALETTE["ink"], anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">'
        f"{escape(value)}</text>"
    )


def save_svg(path: Path, svg: str) -> None:
    path.write_text(svg, encoding="utf-8")


def bar_chart_status(df: pd.DataFrame) -> None:
    counts = df["status"].value_counts().reindex(["operating", "acquired", "closed"])
    total = counts.sum()
    width, height = 900, 300
    left, top, bar_h = 170, 70, 44
    max_v = counts.max()
    parts = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(24, 34, "Current company status is highly censored", 20, 700),
        text(24, 56, "Most companies are still operating, so exit comparisons should focus on mature cohorts.", 13, 400, PALETTE["muted"]),
    ]
    for i, (status, value) in enumerate(counts.items()):
        y = top + i * 64
        w = 610 * value / max_v
        parts.append(text(24, y + 29, status.title(), 13, 700))
        parts.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="6" fill="{PALETTE[status]}"/>')
        parts.append(text(left + w + 14, y + 29, f"{value:,} ({pct(value / total)})", 13, 700))
    save_svg(FIGURES / "status_composition.svg", svg_frame(width, height, "\n".join(parts)))


def cohort_chart(df: pd.DataFrame) -> None:
    cohort = (
        df.dropna(subset=["cohort"])
        .groupby(["cohort", "status"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["operating", "acquired", "closed"], fill_value=0)
    )
    props = cohort.div(cohort.sum(axis=1), axis=0)
    width, height = 920, 380
    left, top, chart_w, bar_h = 120, 80, 680, 36
    parts = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(24, 34, "Outcome mix changes mainly because cohorts have different time to mature", 20, 700),
        text(24, 56, "Later cohorts look healthier because many firms have not yet reached acquisition or closure.", 13, 400, PALETTE["muted"]),
    ]
    for i, cohort_name in enumerate(props.index.astype(str)):
        y = top + i * 54
        x = left
        parts.append(text(24, y + 24, cohort_name, 13, 700))
        for status in ["operating", "acquired", "closed"]:
            w = chart_w * props.loc[cohort_name, status]
            parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="{PALETTE[status]}"/>')
            x += w
        parts.append(text(left + chart_w + 14, y + 24, f"n={int(cohort.loc[cohort_name].sum()):,}", 12, 700, PALETTE["muted"]))
    legend_x = 120
    for status in ["operating", "acquired", "closed"]:
        parts.append(f'<rect x="{legend_x}" y="342" width="14" height="14" rx="3" fill="{PALETTE[status]}"/>')
        parts.append(text(legend_x + 20, 354, status.title(), 12, 700))
        legend_x += 120
    save_svg(FIGURES / "cohort_outcome_mix.svg", svg_frame(width, height, "\n".join(parts)))


def funding_ladder(df: pd.DataFrame) -> None:
    summary = (
        df[df["funding_total_usd"] > 0]
        .dropna(subset=["round_bucket"])
        .groupby(["round_bucket", "status"], observed=False)["funding_total_usd"]
        .median()
        .unstack()
        .reindex(index=["1", "2-3", "4-6", "7+"], columns=["operating", "acquired", "closed"])
    )
    width, height = 920, 430
    left, top, chart_w, chart_h = 90, 78, 760, 260
    values = np.log10(summary.fillna(1).values)
    lo, hi = 4.5, np.nanmax(values) + 0.3
    parts = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(24, 34, "Funding depth separates outcomes, but does not prove causality", 20, 700),
        text(24, 56, "Median total funding rises with follow-on rounds; acquired firms sit highest in most buckets.", 13, 400, PALETTE["muted"]),
    ]
    for tick in [5, 6, 7, 8, 9]:
        y = top + chart_h - (tick - lo) / (hi - lo) * chart_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="{PALETTE["grid"]}" stroke-width="1"/>')
        parts.append(text(24, y + 4, money(10**tick), 11, 400, PALETTE["muted"]))
    group_w = chart_w / len(summary.index)
    bar_w = 44
    offsets = [-52, 0, 52]
    for gi, bucket in enumerate(summary.index.astype(str)):
        cx = left + group_w * gi + group_w / 2
        parts.append(text(cx, top + chart_h + 34, f"{bucket} rounds", 12, 700, anchor="middle"))
        for si, status in enumerate(["operating", "acquired", "closed"]):
            value = summary.loc[bucket, status]
            if pd.isna(value):
                continue
            bar_top = top + chart_h - (np.log10(value) - lo) / (hi - lo) * chart_h
            parts.append(
                f'<rect x="{cx + offsets[si] - bar_w / 2:.1f}" y="{bar_top:.1f}" '
                f'width="{bar_w}" height="{top + chart_h - bar_top:.1f}" rx="5" fill="{PALETTE[status]}"/>'
            )
    legend_x = 260
    for status in ["operating", "acquired", "closed"]:
        parts.append(f'<rect x="{legend_x}" y="390" width="14" height="14" rx="3" fill="{PALETTE[status]}"/>')
        parts.append(text(legend_x + 20, 402, status.title(), 12, 700))
        legend_x += 130
    save_svg(FIGURES / "funding_ladder.svg", svg_frame(width, height, "\n".join(parts)))


def market_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    top = df[df["market"] != "Unknown"]["market"].value_counts().head(15).index
    score = (
        df[df["market"].isin(top)]
        .groupby("market")
        .agg(
            n=("name", "size"),
            acquired_rate=("status", lambda x: (x == "acquired").mean()),
            closed_rate=("status", lambda x: (x == "closed").mean()),
            median_funding=("funding_total_usd", "median"),
        )
        .assign(exit_balance=lambda x: x["acquired_rate"] - x["closed_rate"])
        .sort_values("exit_balance", ascending=False)
    )
    return score


def market_scatter(df: pd.DataFrame) -> pd.DataFrame:
    score = market_scorecard(df)
    width, height = 920, 520
    left, top, chart_w, chart_h = 90, 88, 720, 330
    max_n = score["n"].max()
    max_x = max(0.16, score["closed_rate"].max() * 1.15)
    max_y = max(0.16, score["acquired_rate"].max() * 1.15)
    parts = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(24, 34, "Markets have different exit fingerprints", 20, 700),
        text(24, 56, "Bubble size is company count; stronger markets sit higher and further left.", 13, 400, PALETTE["muted"]),
        f'<rect x="{left}" y="{top}" width="{chart_w}" height="{chart_h}" fill="{PALETTE["surface"]}" stroke="{PALETTE["grid"]}"/>',
    ]
    for t in np.linspace(0, max_x, 5):
        x = left + chart_w * t / max_x
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + chart_h}" stroke="{PALETTE["grid"]}" stroke-width="1"/>')
        parts.append(text(x, top + chart_h + 24, pct(t), 11, 400, PALETTE["muted"], "middle"))
    for t in np.linspace(0, max_y, 5):
        y = top + chart_h - chart_h * t / max_y
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="{PALETTE["grid"]}" stroke-width="1"/>')
        parts.append(text(left - 12, y + 4, pct(t), 11, 400, PALETTE["muted"], "end"))
    parts.append(text(left + chart_w / 2, height - 32, "Closure rate", 13, 700, anchor="middle"))
    parts.append(text(24, top + chart_h / 2, "Acquisition rate", 13, 700, anchor="start"))
    for market, row in score.iterrows():
        x = left + chart_w * row["closed_rate"] / max_x
        y = top + chart_h - chart_h * row["acquired_rate"] / max_y
        r = 7 + 18 * np.sqrt(row["n"] / max_n)
        color = PALETTE["acquired"] if row["exit_balance"] >= 0 else PALETTE["closed"]
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" fill-opacity="0.70" stroke="white" stroke-width="2"/>')
        if market in ["Software", "Biotechnology", "Enterprise Software", "Curated Web", "Games", "Mobile"]:
            parts.append(text(x + r + 4, y + 4, market, 11, 700))
    save_svg(FIGURES / "market_exit_fingerprint.svg", svg_frame(width, height, "\n".join(parts)))
    return score


def formation_line(df: pd.DataFrame) -> None:
    counts = df.dropna(subset=["founded_year"]).query("founded_year >= 1980 and founded_year <= 2014").groupby("founded_year").size()
    width, height = 900, 360
    left, top, chart_w, chart_h = 80, 70, 760, 220
    xs = counts.index.to_numpy()
    ys = counts.to_numpy()
    xscale = lambda x: left + (x - xs.min()) / (xs.max() - xs.min()) * chart_w
    yscale = lambda y: top + chart_h - y / ys.max() * chart_h
    points = " ".join(f"{xscale(x):.1f},{yscale(y):.1f}" for x, y in zip(xs, ys))
    parts = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        text(24, 34, "The dataset is dominated by the platform-startup boom", 20, 700),
        text(24, 56, "Company formation accelerates after 2000, then drops near the dataset boundary.", 13, 400, PALETTE["muted"]),
        f'<polyline points="{points}" fill="none" stroke="{PALETTE["accent"]}" stroke-width="4" stroke-linejoin="round"/>',
    ]
    for year in [1980, 1990, 2000, 2010, 2014]:
        x = xscale(year)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + chart_h}" stroke="{PALETTE["grid"]}" stroke-width="1"/>')
        parts.append(text(x, top + chart_h + 25, str(year), 11, 400, PALETTE["muted"], "middle"))
    for val in [0, 1500, 3000, 4500, 6000]:
        y = yscale(val)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="{PALETTE["grid"]}" stroke-width="1"/>')
        parts.append(text(left - 10, y + 4, f"{val:,}", 11, 400, PALETTE["muted"], "end"))
    save_svg(FIGURES / "formation_curve.svg", svg_frame(width, height, "\n".join(parts)))


def table_html(df: pd.DataFrame) -> str:
    score = market_scorecard(df).head(10).copy()
    rows = []
    for market, row in score.iterrows():
        rows.append(
            "<tr>"
            f"<td>{escape(market)}</td>"
            f"<td>{int(row['n']):,}</td>"
            f"<td>{pct(row['acquired_rate'])}</td>"
            f"<td>{pct(row['closed_rate'])}</td>"
            f"<td>{money(row['median_funding'])}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def write_report(df: pd.DataFrame) -> None:
    n = len(df)
    markets = df["market"].nunique()
    countries = df["country_code"].nunique()
    median_funding = df.loc[df["funding_total_usd"] > 0, "funding_total_usd"].median()
    mature = df[df["founded_year"].between(2000, 2008)]
    acquired = (mature["status"] == "acquired").mean()
    closed = (mature["status"] == "closed").mean()
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Venture Outcomes Under Censoring</title>
  <style>
    :root {{ --ink:#172033; --muted:#5d6472; --line:#d9dee8; --paper:#ffffff; --wash:#f5f7fb; --accent:#3457d5; }}
    body {{ margin:0; font-family: Inter, Arial, sans-serif; color:var(--ink); background:var(--wash); line-height:1.65; }}
    main {{ max-width:1100px; margin:0 auto; padding:48px 20px 72px; }}
    header {{ padding:40px 0 26px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0; font-size:44px; line-height:1.08; letter-spacing:0; }}
    h2 {{ margin:42px 0 12px; font-size:28px; letter-spacing:0; }}
    h3 {{ margin:28px 0 8px; font-size:19px; }}
    p {{ max-width:860px; }}
    a {{ color:var(--accent); font-weight:700; }}
    .dek {{ color:var(--muted); font-size:18px; max-width:900px; }}
    .meta {{ color:var(--muted); font-size:14px; text-transform:uppercase; letter-spacing:.08em; font-weight:800; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:28px 0; }}
    .stat {{ background:var(--paper); border:1px solid var(--line); padding:18px; border-radius:10px; }}
    .stat b {{ display:block; font-size:26px; }}
    .stat span {{ color:var(--muted); font-size:13px; }}
    .panel {{ background:var(--paper); border:1px solid var(--line); padding:18px; border-radius:12px; margin:18px 0; }}
    .panel img {{ width:100%; height:auto; display:block; }}
    table {{ width:100%; border-collapse:collapse; background:var(--paper); border:1px solid var(--line); }}
    th,td {{ padding:11px 12px; border-bottom:1px solid var(--line); text-align:left; }}
    th {{ background:#eef2fb; font-size:13px; text-transform:uppercase; letter-spacing:.06em; }}
    .callout {{ border-left:5px solid var(--accent); background:var(--paper); padding:16px 18px; margin:22px 0; }}
    .small {{ color:var(--muted); font-size:14px; }}
    @media (max-width:760px) {{ h1 {{ font-size:34px; }} .grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
  </style>
</head>
<body>
<main>
  <header>
    <div class="meta">Publication-style EDA / Crunchbase venture dataset</div>
    <h1>Venture Outcomes Under Censoring</h1>
    <p class="dek">A descriptive study of how startup status, funding depth, founding cohort, and market structure interact in a static Crunchbase investment snapshot. The central point is not that funding mechanically causes success; it is that startup data must be read through time-to-outcome and survivorship bias.</p>
  </header>

  <section class="grid">
    <div class="stat"><b>{n:,}</b><span>companies with known status</span></div>
    <div class="stat"><b>{markets:,}</b><span>market labels</span></div>
    <div class="stat"><b>{countries:,}</b><span>countries represented</span></div>
    <div class="stat"><b>{money(median_funding)}</b><span>median disclosed funding</span></div>
  </section>

  <section>
    <h2>Executive Findings</h2>
    <div class="callout">
      <b>Best interpretation:</b> funding depth is a marker of startup selection and maturity, while company status is heavily censored because most recent firms are still operating.
    </div>
    <p>In mature 2000-2008 founding cohorts, {pct(acquired)} of companies are marked acquired and {pct(closed)} are marked closed. Later cohorts should not be compared directly because many companies have not had enough time to reach an exit or failure event.</p>
  </section>

  <section>
    <h2>Dataset Audit</h2>
    <p>The file contains firm-level Crunchbase records: market, geography, founding year, total disclosed funding, funding rounds, and current status. It is a useful public dataset for portfolio analysis, but it is not a complete population of all startups. It overrepresents companies visible to venture databases and underrepresents unreported small firms.</p>
    <div class="panel"><img src="figures/status_composition.svg" alt="Status composition chart"></div>
  </section>

  <section>
    <h2>Cohort Effects</h2>
    <p>Operating status dominates because the dataset contains many young firms. For this reason, outcome analysis is most meaningful when founding year is treated as a proxy for time at risk.</p>
    <div class="panel"><img src="figures/formation_curve.svg" alt="Startup formation curve"></div>
    <div class="panel"><img src="figures/cohort_outcome_mix.svg" alt="Outcome mix by founding cohort"></div>
  </section>

  <section>
    <h2>Funding Depth</h2>
    <p>The funding ladder shows a consistent descriptive pattern: companies with more rounds have higher median disclosed funding, and acquired firms tend to sit above operating and closed firms within comparable round buckets. This is selection, not causal proof.</p>
    <div class="panel"><img src="figures/funding_ladder.svg" alt="Funding ladder by status and rounds"></div>
  </section>

  <section>
    <h2>Market Exit Fingerprints</h2>
    <p>Market categories differ in both acquisition and closure rates. Software is large, but not automatically the most favorable by outcome mix. Enterprise Software and Advertising show relatively stronger acquisition balance among high-volume categories, while Curated Web and Games show comparatively higher closure pressure.</p>
    <div class="panel"><img src="figures/market_exit_fingerprint.svg" alt="Market acquisition and closure scatterplot"></div>
    <table>
      <thead><tr><th>Market</th><th>Companies</th><th>Acquired</th><th>Closed</th><th>Median Funding</th></tr></thead>
      <tbody>{table_html(df)}</tbody>
    </table>
  </section>

  <section>
    <h2>Limitations</h2>
    <p>This report makes descriptive claims only. The dataset is static, status labels are censored, funding is aggregated rather than transactional, and important features such as founder background, revenue, investor quality, burn rate, and product traction are absent. A stronger causal study would require time-stamped financing events and survival modeling.</p>
    <p class="small">Generated from <code>scripts/build_publication_report.py</code>. Source dataset: <code>investments_VC.csv</code>.</p>
  </section>
</main>
</body>
</html>
"""
    REPORT.write_text(html, encoding="utf-8")


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    df = read_data()
    bar_chart_status(df)
    cohort_chart(df)
    funding_ladder(df)
    formation_line(df)
    market_scatter(df)
    write_report(df)


if __name__ == "__main__":
    main()
