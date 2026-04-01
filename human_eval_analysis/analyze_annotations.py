"""
Inter-annotator agreement and descriptive statistics for human evaluation annotations.

Generates a markdown report (analysis_report.md) with:
  - Dimension score agreement (Krippendorff's alpha, ICC, Spearman, Cohen's kappa)
  - Attribution agreement
  - Descriptive statistics and annotator bias
  - Dimension correlations
  - Per-episode disagreement analysis
  - Per-model score breakdowns (if episode_models.json exists)
"""

from __future__ import annotations

import io
import json
import warnings
from itertools import combinations
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

warnings.filterwarnings("ignore", category=RuntimeWarning)

ANALYSIS_DIR = Path(__file__).resolve().parent
ANNOTATIONS_DIR = ANALYSIS_DIR / "annotations"
MODELS_PATH = ANALYSIS_DIR / "episode_models.json"
REPORT_PATH = ANALYSIS_DIR / "analysis_report.md"
REPORT_COMBINED_PATH = ANALYSIS_DIR / "analysis_report_combined.md"
DATA_DIR = ANALYSIS_DIR.parent / "data"

DIMENSIONS = ["goal", "believability", "knowledge", "secret", "relationship", "social_rules", "financial"]

HUMAN_TO_AI_DIM = {
    "goal": "goal", "believability": "believability", "knowledge": "knowledge",
    "secret": "secret", "relationship": "relationship", "social_rules": "social_rules",
    "financial": "financial_and_material_benefits",
}

ANNOTATOR_FOLDERS = {
    "an01_split_1": "split_1",
    "an01_split_2": "split_2",
    "an02_split_1": "split_1",
    "an02_split_2": "split_2",
    "an03_split_1": "split_1",
}

# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------

class MarkdownWriter:
    def __init__(self):
        self._buf = io.StringIO()

    def h1(self, text: str):
        self._buf.write(f"\n# {text}\n\n")

    def h2(self, text: str):
        self._buf.write(f"\n## {text}\n\n")

    def h3(self, text: str):
        self._buf.write(f"\n### {text}\n\n")

    def p(self, text: str):
        self._buf.write(f"{text}\n\n")

    def bullet(self, text: str):
        self._buf.write(f"- {text}\n")

    def end_bullets(self):
        self._buf.write("\n")

    def table(self, rows: list[dict], **kwargs):
        if not rows:
            self._buf.write("*No data.*\n\n")
            return
        self._buf.write(tabulate(rows, headers="keys", tablefmt="github", **kwargs))
        self._buf.write("\n\n")

    def table_df(self, df: pd.DataFrame, **kwargs):
        self._buf.write(tabulate(df, headers="keys", tablefmt="github", showindex=False, **kwargs))
        self._buf.write("\n\n")

    def code_block(self, text: str):
        self._buf.write(f"```\n{text}\n```\n\n")

    def getvalue(self) -> str:
        return self._buf.getvalue()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_episode_models() -> dict:
    if MODELS_PATH.exists():
        with open(MODELS_PATH) as f:
            return json.load(f)
    return {}


def load_all_annotations(episode_models: dict | None = None) -> pd.DataFrame:
    if episode_models is None:
        episode_models = {}
    rows = []
    for folder_name, split in ANNOTATOR_FOLDERS.items():
        folder = ANNOTATIONS_DIR / folder_name
        if not folder.exists():
            continue
        for fp in sorted(folder.glob("*.json")):
            with open(fp) as f:
                ann = json.load(f)
            annotator = ann["annotator"]
            episode_id = ann["episode_id"]
            agent_names = list(next(iter(ann["dimension_scores"].values())).keys())
            model_info = episode_models.get(episode_id, {})
            for dim in DIMENSIONS:
                for agent_idx, agent_name in enumerate(agent_names):
                    score = ann["dimension_scores"].get(dim, {}).get(agent_name)
                    rows.append({
                        "split": split, "episode_id": episode_id, "annotator": annotator,
                        "dimension": dim, "agent_idx": agent_idx + 1,
                        "agent_name": agent_name, "score": score,
                        "model": model_info.get(f"agent_{agent_idx + 1}_model", ""),
                    })
    return pd.DataFrame(rows)


def load_all_attributions(episode_models: dict | None = None) -> pd.DataFrame:
    if episode_models is None:
        episode_models = {}
    rows = []
    for folder_name, split in ANNOTATOR_FOLDERS.items():
        folder = ANNOTATIONS_DIR / folder_name
        if not folder.exists():
            continue
        for fp in sorted(folder.glob("*.json")):
            with open(fp) as f:
                ann = json.load(f)
            annotator = ann["annotator"]
            episode_id = ann["episode_id"]
            model_info = episode_models.get(episode_id, {})
            for dim in DIMENSIONS:
                attr = ann.get("dimension_attributions", {}).get(dim, {})
                for turn_key, value in attr.items():
                    agent_num = turn_key.split("_")[1]
                    rows.append({
                        "split": split, "episode_id": episode_id, "annotator": annotator,
                        "dimension": dim, "turn_key": turn_key, "attribution": value,
                        "model": model_info.get(f"agent_{agent_num}_model", ""),
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------

def compute_krippendorff_alpha(rating_matrix: np.ndarray, level: str = "ordinal") -> float:
    try:
        return krippendorff.alpha(reliability_data=rating_matrix, level_of_measurement=level)
    except Exception:
        return np.nan


def compute_icc(df_wide: pd.DataFrame) -> float:
    df = df_wide.dropna(how="any")
    if df.shape[0] < 2 or df.shape[1] < 2:
        return np.nan
    n, k = df.shape
    grand_mean = df.values.mean()
    row_means = df.values.mean(axis=1)
    col_means = df.values.mean(axis=0)
    ss_total = np.sum((df.values - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1) if n > 1 else 0
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n > 1 and k > 1) else 0
    ms_cols = ss_cols / (k - 1) if k > 1 else 0
    denom = ms_rows + (k - 1) * ms_error / n + k * (ms_cols - ms_error) / n
    if denom == 0:
        return np.nan
    return (ms_rows - ms_error) / denom


def pairwise_spearman(df_wide: pd.DataFrame) -> dict:
    cols = df_wide.columns.tolist()
    results = {}
    for a, b in combinations(cols, 2):
        mask = df_wide[[a, b]].dropna()
        if len(mask) < 3 or mask[a].std() == 0 or mask[b].std() == 0:
            results[(a, b)] = np.nan
            continue
        rho, _ = stats.spearmanr(mask[a], mask[b])
        results[(a, b)] = rho
    return results


def cohens_kappa_pairwise(df_wide: pd.DataFrame) -> dict:
    from sklearn.metrics import cohen_kappa_score
    cols = df_wide.columns.tolist()
    results = {}
    for a, b in combinations(cols, 2):
        mask = df_wide[[a, b]].dropna()
        if len(mask) < 2:
            results[(a, b)] = np.nan
            continue
        if mask[a].nunique() <= 1 and mask[b].nunique() <= 1:
            results[(a, b)] = 1.0 if (mask[a] == mask[b]).all() else np.nan
            continue
        try:
            results[(a, b)] = cohen_kappa_score(mask[a].astype(int), mask[b].astype(int))
        except Exception:
            results[(a, b)] = np.nan
    return results


def fmt(val: float, decimals: int = 4) -> str:
    return f"{val:.{decimals}f}" if not np.isnan(val) else "N/A"


# ---------------------------------------------------------------------------
# Pivot helpers
# ---------------------------------------------------------------------------

def pivot_scores(df: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    sub = df if split is None else df[df["split"] == split]
    return sub.pivot_table(index=["episode_id", "dimension", "agent_idx"], columns="annotator", values="score")


def pivot_attributions(df: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    sub = df if split is None else df[df["split"] == split]
    return sub.pivot_table(index=["episode_id", "dimension", "turn_key"], columns="annotator", values="attribution")


def _bin_attribution(val):
    """Bin attribution: 0 -> 0, (1,2) -> 1, 3 -> 2."""
    if val <= 0:
        return 0
    if val >= 3:
        return 2
    return 1


def pivot_attributions_binned(df: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    """Pivot attributions with 3-level binning: 0, (1-2), 3."""
    sub = df if split is None else df[df["split"] == split]
    sub = sub.copy()
    sub["attribution_binned"] = sub["attribution"].apply(_bin_attribution)
    return sub.pivot_table(index=["episode_id", "dimension", "turn_key"], columns="annotator", values="attribution_binned")


# ---------------------------------------------------------------------------
# Agreement reporting
# ---------------------------------------------------------------------------

def report_agreement(md: MarkdownWriter, pivot_df: pd.DataFrame):
    annotators = pivot_df.columns.tolist()
    n_items = len(pivot_df)
    n_complete = pivot_df.dropna(how="any").shape[0]

    md.bullet(f"**Annotators:** {', '.join(annotators)}")
    md.bullet(f"**Items:** {n_items} (complete cases: {n_complete})")

    alpha = compute_krippendorff_alpha(pivot_df.values.T)
    md.bullet(f"**Krippendorff's alpha (ordinal):** {fmt(alpha)}")

    icc = compute_icc(pivot_df)
    md.bullet(f"**ICC(2,1):** {fmt(icc)}")

    sp = pairwise_spearman(pivot_df)
    if sp:
        vals = [v for v in sp.values() if not np.isnan(v)]
        mean_rho = np.mean(vals) if vals else np.nan
        md.bullet(f"**Mean pairwise Spearman rho:** {fmt(mean_rho)}")
        for pair, rho in sp.items():
            md.bullet(f"  {pair[0]} vs {pair[1]}: {fmt(rho)}")

    kappas = cohens_kappa_pairwise(pivot_df)
    if kappas:
        vals = [v for v in kappas.values() if not np.isnan(v)]
        mean_k = np.mean(vals) if vals else np.nan
        md.bullet(f"**Mean pairwise Cohen's kappa:** {fmt(mean_k)}")
        for pair, k in kappas.items():
            md.bullet(f"  {pair[0]} vs {pair[1]}: {fmt(k)}")
    md.end_bullets()


def agreement_per_dim_table(pivot_fn, data: pd.DataFrame, show_kappa: bool = False) -> list[dict]:
    rows = []
    for dim in DIMENSIONS:
        piv_dim = pivot_fn(data[data["dimension"] == dim])
        if piv_dim.empty:
            continue
        alpha = compute_krippendorff_alpha(piv_dim.values.T)
        icc = compute_icc(piv_dim)
        sp = pairwise_spearman(piv_dim)
        sp_vals = [v for v in sp.values() if not np.isnan(v)]
        mean_rho = np.mean(sp_vals) if sp_vals else np.nan
        row = {"Dimension": dim, "K-alpha": fmt(alpha, 3), "ICC(2,1)": fmt(icc, 3),
               "Mean Spearman": fmt(mean_rho, 3), "N items": len(piv_dim)}
        if show_kappa:
            kappas = cohens_kappa_pairwise(piv_dim)
            k_vals = [v for v in kappas.values() if not np.isnan(v)]
            mean_k = np.mean(k_vals) if k_vals else np.nan
            row["Mean Kappa"] = fmt(mean_k, 3)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_episode_annotator_counts(scores_df: pd.DataFrame) -> pd.DataFrame:
    return (
        scores_df.groupby(["split", "episode_id"])["annotator"].nunique()
        .reset_index().rename(columns={"annotator": "n_annotators"})
    )


def filter_by_annotator_count(df: pd.DataFrame, counts: pd.DataFrame, min_count: int) -> pd.DataFrame:
    valid = counts[counts["n_annotators"] >= min_count][["split", "episode_id"]]
    return df.merge(valid, on=["split", "episode_id"], how="inner")


# ---------------------------------------------------------------------------
# AI annotation loading
# ---------------------------------------------------------------------------

def _parse_ai_file(ai_file: Path, human_dim: str, run: int) -> tuple[list[dict], list[dict]]:
    """Parse a single AI annotation JSONL file into score and attribution rows."""
    import re as _re
    score_rows, attr_rows = [], []
    with open(ai_file) as f:
        for line in f:
            rec = json.loads(line)
            episode_id = rec["episode_id"]
            agent_name = rec["agent"]
            agent_idx = 1 if rec["is_first_speaker"] else 2

            for utt_key, utt_val in rec["attributed_utterances"].items():
                if agent_name not in utt_key:
                    continue
                if isinstance(utt_val[1], dict):
                    attr_score = utt_val[1].get("attribution", 0)
                    dim_score = utt_val[1].get("dim_score", None)
                elif isinstance(utt_val[1], (int, float)):
                    attr_score = utt_val[1]
                    dim_score = None
                else:
                    continue

                m = _re.match(r"Utterance (\d+) by .+", utt_key)
                if not m:
                    continue
                local_turn = int(m.group(1))
                turn_key = f"agent_{agent_idx}_turn_{local_turn}"

                attr_rows.append({
                    "episode_id": episode_id, "dimension": human_dim,
                    "turn_key": turn_key, "ai_attribution": attr_score, "run": run,
                })

                if dim_score is not None and local_turn == 0:
                    score_rows.append({
                        "episode_id": episode_id, "dimension": human_dim,
                        "agent_idx": agent_idx, "agent_name": agent_name,
                        "ai_score": dim_score, "run": run,
                    })
    return score_rows, attr_rows


def load_ai_annotations() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load LLM attribution annotations from all runs.

    Discovers files matching human_eval_annotated_default-{dim}_gpt-4o_run*.jsonl.
    Returns per-run data with a 'run' column. Aggregation is done at analysis time.
    """
    import glob as _glob
    score_rows = []
    attr_rows = []

    for human_dim, ai_dim in HUMAN_TO_AI_DIM.items():
        pattern = str(DATA_DIR / f"human_eval_annotated_default-{ai_dim}_gpt-4o_run*.jsonl")
        files = sorted(_glob.glob(pattern))
        if not files:
            old = DATA_DIR / f"human_eval_annotated_default-{ai_dim}_gpt-4o.jsonl"
            if old.exists():
                files = [str(old)]
        for fpath in files:
            fname = Path(fpath).stem
            if "_run" in fname:
                run = int(fname.rsplit("_run", 1)[1])
            else:
                run = 1
            s, a = _parse_ai_file(Path(fpath), human_dim, run)
            score_rows.extend(s)
            attr_rows.extend(a)

    scores_df = pd.DataFrame(score_rows) if score_rows else pd.DataFrame()
    attr_df = pd.DataFrame(attr_rows) if attr_rows else pd.DataFrame()

    if not scores_df.empty:
        n_runs = scores_df["run"].nunique()
        print(f"  AI annotations: {n_runs} run(s), {len(scores_df)} score rows, {len(attr_df)} attr rows")

    return scores_df, attr_df


def _score_alignment_table(merged: pd.DataFrame, dims: list[str], label_h: str = "human_score", label_a: str = "ai_score") -> list[dict]:
    """Build per-dimension + overall alignment table rows."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    rows = []
    for dim in dims:
        sub = merged[merged["dimension"] == dim]
        if len(sub) < 3:
            continue
        h, a = sub[label_h].values, sub[label_a].values
        rho = _safe_spearman(h, a)
        pearson_r = _safe_pearson(h, a)
        rows.append({
            "Dimension": dim, "N": len(sub),
            "Spearman ρ": fmt(rho, 3), "Pearson r": fmt(pearson_r, 3),
            "MAE": fmt(mean_absolute_error(h, a), 2),
            "RMSE": fmt(np.sqrt(mean_squared_error(h, a)), 2),
            "AI bias": fmt(np.mean(a - h), 2),
            "Human mean": fmt(np.mean(h), 2), "AI mean": fmt(np.mean(a), 2),
        })
    h_all, a_all = merged[label_h].values, merged[label_a].values
    rho_all = _safe_spearman(h_all, a_all)
    pearson_all = _safe_pearson(h_all, a_all)
    rows.append({
        "Dimension": "**OVERALL**", "N": len(h_all),
        "Spearman ρ": fmt(rho_all, 3), "Pearson r": fmt(pearson_all, 3),
        "MAE": fmt(mean_absolute_error(h_all, a_all), 2),
        "RMSE": fmt(np.sqrt(mean_squared_error(h_all, a_all)), 2),
        "AI bias": fmt(np.mean(a_all - h_all), 2),
        "Human mean": fmt(np.mean(h_all), 2), "AI mean": fmt(np.mean(a_all), 2),
    })
    return rows


def _safe_spearman(h, a):
    if len(h) < 3 or np.std(h) == 0 or np.std(a) == 0:
        return np.nan
    return stats.spearmanr(h, a)[0]


def _safe_pearson(h, a):
    if len(h) < 3 or np.std(h) == 0 or np.std(a) == 0:
        return np.nan
    return stats.pearsonr(h, a)[0]


def _attr_alignment_table(merged: pd.DataFrame, dims: list[str]) -> list[dict]:
    """Build per-dimension + overall attribution alignment table rows."""
    from sklearn.metrics import cohen_kappa_score, mean_absolute_error
    rows = []
    for dim in dims:
        sub = merged[merged["dimension"] == dim]
        if len(sub) < 3:
            continue
        h, a = sub["human_attr"].values, sub["ai_attribution"].values
        rho = _safe_spearman(h, a)
        pearson_r = _safe_pearson(h, a)
        mae = mean_absolute_error(h, a)
        nz_agree = np.mean((h > 0) == (a > 0)) * 100 if len(h) > 0 else 0
        try:
            h_bin, a_bin = (h > 0).astype(int), (a > 0).astype(int)
            kappa = cohen_kappa_score(h_bin, a_bin) if (len(np.unique(h_bin)) > 1 or len(np.unique(a_bin)) > 1) else np.nan
        except Exception:
            kappa = np.nan
        rows.append({
            "Dimension": dim, "N": len(sub),
            "Spearman ρ": fmt(rho, 3), "Pearson r": fmt(pearson_r, 3),
            "MAE": fmt(mae, 2), "Zero/NZ agree %": f"{nz_agree:.1f}%",
            "Binary κ": fmt(kappa, 3),
            "Human mean": fmt(np.mean(h), 2), "AI mean": fmt(np.mean(a), 2),
        })
    h_all, a_all = merged["human_attr"].values, merged["ai_attribution"].values
    rho_all = _safe_spearman(h_all, a_all)
    pearson_all = _safe_pearson(h_all, a_all)
    nz_all = np.mean((h_all > 0) == (a_all > 0)) * 100
    rows.append({
        "Dimension": "**OVERALL**", "N": len(h_all),
        "Spearman ρ": fmt(rho_all, 3), "Pearson r": fmt(pearson_all, 3),
        "MAE": fmt(mean_absolute_error(h_all, a_all), 2),
        "Zero/NZ agree %": f"{nz_all:.1f}%", "Binary κ": "—",
        "Human mean": fmt(np.mean(h_all), 2), "AI mean": fmt(np.mean(a_all), 2),
    })
    return rows


def human_ai_alignment_analysis(
    md: MarkdownWriter,
    human_scores: pd.DataFrame,
    human_attr: pd.DataFrame,
    ai_scores: pd.DataFrame,
    ai_attr: pd.DataFrame,
):
    """Comprehensive human vs AI alignment analysis with multiple aggregation strategies."""
    from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error

    n_ai_runs = ai_scores["run"].nunique() if "run" in ai_scores.columns else 1
    multi_run = n_ai_runs > 1

    # --- Aggregate AI across runs ---
    def _agg_ai_scores(agg: str = "mean") -> pd.DataFrame:
        if "run" not in ai_scores.columns or n_ai_runs == 1:
            return ai_scores[["episode_id", "dimension", "agent_idx", "ai_score"]].drop_duplicates()
        return (
            ai_scores.groupby(["episode_id", "dimension", "agent_idx"])["ai_score"]
            .agg(agg).reset_index()
        )

    def _agg_ai_attr(agg: str = "mean") -> pd.DataFrame:
        if "run" not in ai_attr.columns or n_ai_runs == 1:
            return ai_attr[["episode_id", "dimension", "turn_key", "ai_attribution"]].drop_duplicates()
        return (
            ai_attr.groupby(["episode_id", "dimension", "turn_key"])["ai_attribution"]
            .agg(agg).reset_index()
        )

    # --- Merge helpers ---
    def _merge_scores(h_scores: pd.DataFrame, h_agg: str = "mean", ai_agg: str = "mean") -> pd.DataFrame:
        human_agg = (
            h_scores.groupby(["episode_id", "dimension", "agent_idx"])["score"]
            .agg(h_agg).reset_index().rename(columns={"score": "human_score"})
        )
        return human_agg.merge(_agg_ai_scores(ai_agg), on=["episode_id", "dimension", "agent_idx"], how="inner")

    def _merge_attr(h_attr: pd.DataFrame, h_agg: str = "mean", ai_agg: str = "mean") -> pd.DataFrame:
        human_agg = (
            h_attr.groupby(["episode_id", "dimension", "turn_key"])["attribution"]
            .agg(h_agg).reset_index().rename(columns={"attribution": "human_attr"})
        )
        return human_agg.merge(_agg_ai_attr(ai_agg), on=["episode_id", "dimension", "turn_key"], how="inner")

    # --- Annotator count helpers ---
    ep_ann_counts = human_scores.groupby("episode_id")["annotator"].nunique().reset_index()
    ep_ann_counts.columns = ["episode_id", "n_ann"]
    eps_2plus = set(ep_ann_counts[ep_ann_counts["n_ann"] >= 2]["episode_id"])
    eps_3 = set(ep_ann_counts[ep_ann_counts["n_ann"] >= 3]["episode_id"])

    # --- Build all merged datasets (single-run AI = run 1 baseline) ---
    merged_scores = _merge_scores(human_scores, "mean", "mean")
    merged_scores_median = _merge_scores(human_scores, "median", "mean")
    merged_scores_2ann = _merge_scores(human_scores[human_scores["episode_id"].isin(eps_2plus)], "mean", "mean")
    merged_scores_3ann = _merge_scores(human_scores[human_scores["episode_id"].isin(eps_3)], "mean", "mean")

    # Bias-corrected
    dim_bias = merged_scores.groupby("dimension").apply(
        lambda g: np.mean(g["ai_score"].values - g["human_score"].values)
    ).to_dict()
    merged_scores_bc = merged_scores.copy()
    merged_scores_bc["ai_score"] = merged_scores_bc.apply(
        lambda r: r["ai_score"] - dim_bias.get(r["dimension"], 0), axis=1
    )

    # Multi-run AI aggregated scores (if multiple runs available)
    if multi_run:
        merged_scores_ai_mean = _merge_scores(human_scores, "mean", "mean")
        merged_scores_ai_median = _merge_scores(human_scores, "mean", "median")
        merged_scores_hmed_aimed = _merge_scores(human_scores, "median", "median")
        merged_scores_3ann_ai_mean = _merge_scores(
            human_scores[human_scores["episode_id"].isin(eps_3)], "mean", "mean")
        merged_scores_3ann_ai_median = _merge_scores(
            human_scores[human_scores["episode_id"].isin(eps_3)], "mean", "median")

    # Attributions
    merged_attr = _merge_attr(human_attr, "mean", "mean")
    merged_attr_median = _merge_attr(human_attr, "median", "mean")
    merged_attr_3ann = _merge_attr(human_attr[human_attr["episode_id"].isin(eps_3)], "mean", "mean")
    merged_attr_3ann_median = _merge_attr(human_attr[human_attr["episode_id"].isin(eps_3)], "median", "mean")

    # Multi-run AI aggregated attributions
    if multi_run:
        merged_attr_ai_mean = _merge_attr(human_attr, "mean", "mean")
        merged_attr_ai_median = _merge_attr(human_attr, "mean", "median")
        merged_attr_hmed_aimed = _merge_attr(human_attr, "median", "median")
        merged_attr_3ann_ai_mean = _merge_attr(
            human_attr[human_attr["episode_id"].isin(eps_3)], "mean", "mean")
        merged_attr_3ann_ai_median = _merge_attr(
            human_attr[human_attr["episode_id"].isin(eps_3)], "mean", "median")

    # Within-episode rank correlation
    def _within_episode_rank_corr(merged_a: pd.DataFrame, dims: list[str]) -> list[dict]:
        rows = []
        for dim in dims:
            sub = merged_a[merged_a["dimension"] == dim]
            rhos = []
            for eid in sub["episode_id"].unique():
                ep = sub[sub["episode_id"] == eid]
                if len(ep) < 3:
                    continue
                h, a = ep["human_attr"].values, ep["ai_attribution"].values
                if np.std(h) > 0 and np.std(a) > 0:
                    rhos.append(stats.spearmanr(h, a)[0])
            rows.append({
                "Dimension": dim, "N episodes": len(rhos),
                "Mean within-ep ρ": fmt(np.mean(rhos), 3) if rhos else "N/A",
                "Median within-ep ρ": fmt(np.median(rhos), 3) if rhos else "N/A",
            })
        all_rhos = []
        for eid in merged_a["episode_id"].unique():
            ep = merged_a[merged_a["episode_id"] == eid]
            if len(ep) < 3:
                continue
            h, a = ep["human_attr"].values, ep["ai_attribution"].values
            if np.std(h) > 0 and np.std(a) > 0:
                all_rhos.append(stats.spearmanr(h, a)[0])
        rows.append({
            "Dimension": "**ALL DIMS**", "N episodes": len(all_rhos),
            "Mean within-ep ρ": fmt(np.mean(all_rhos), 3) if all_rhos else "N/A",
            "Median within-ep ρ": fmt(np.median(all_rhos), 3) if all_rhos else "N/A",
        })
        return rows

    # Per-episode rank-normalized attribution correlation
    def _rank_normalized_attr_table(merged_a: pd.DataFrame, dims: list[str]) -> list[dict]:
        """Rank-normalize attributions within each (episode, dimension) before correlating."""
        from scipy.stats import rankdata
        rows_out = []
        for dim in dims:
            sub = merged_a[merged_a["dimension"] == dim].copy()
            h_ranks, a_ranks = [], []
            for eid in sub["episode_id"].unique():
                ep = sub[sub["episode_id"] == eid]
                if len(ep) < 2:
                    continue
                h_ranks.extend(rankdata(ep["human_attr"].values))
                a_ranks.extend(rankdata(ep["ai_attribution"].values))
            h_ranks, a_ranks = np.array(h_ranks), np.array(a_ranks)
            rho = _safe_spearman(h_ranks, a_ranks)
            pearson_r = _safe_pearson(h_ranks, a_ranks)
            rows_out.append({
                "Dimension": dim, "N": len(h_ranks),
                "Spearman ρ (ranked)": fmt(rho, 3), "Pearson r (ranked)": fmt(pearson_r, 3),
            })
        all_h, all_a = [], []
        for eid in merged_a["episode_id"].unique():
            ep = merged_a[merged_a["episode_id"] == eid]
            if len(ep) < 2:
                continue
            all_h.extend(rankdata(ep["human_attr"].values))
            all_a.extend(rankdata(ep["ai_attribution"].values))
        all_h, all_a = np.array(all_h), np.array(all_a)
        rows_out.append({
            "Dimension": "**OVERALL**", "N": len(all_h),
            "Spearman ρ (ranked)": fmt(_safe_spearman(all_h, all_a), 3),
            "Pearson r (ranked)": fmt(_safe_pearson(all_h, all_a), 3),
        })
        return rows_out

    n_episodes = merged_scores["episode_id"].nunique()
    md.h2("16. Human vs AI Alignment — Overview")
    md.p(f"**Matched episodes:** {n_episodes} | **Score pairs:** {len(merged_scores)} | **Attribution pairs:** {len(merged_attr)}")
    md.p(f"**Episodes with ≥2 annotators:** {len(eps_2plus)} | **Episodes with 3 annotators:** {len(eps_3)}")
    md.p(f"**AI annotation runs:** {n_ai_runs}")
    md.p("Multiple aggregation strategies are compared below (each is additive, not a replacement).")

    # =====================================================================
    # SCORES
    # =====================================================================
    md.h2("17. Human vs AI — Dimension Score Alignment")

    md.h3("A. Human mean, all annotators (baseline)")
    md.table(_score_alignment_table(merged_scores, DIMENSIONS))

    md.h3("B. Human median (robust to outlier annotators)")
    md.table(_score_alignment_table(merged_scores_median, DIMENSIONS))

    md.h3("C. Only ≥2 annotator episodes")
    md.p(f"Episodes: {merged_scores_2ann['episode_id'].nunique()}")
    md.table(_score_alignment_table(merged_scores_2ann, DIMENSIONS))

    md.h3("D. Only 3-annotator episodes (highest-confidence consensus)")
    md.p(f"Episodes: {merged_scores_3ann['episode_id'].nunique()}")
    md.table(_score_alignment_table(merged_scores_3ann, DIMENSIONS))

    md.h3("E. Bias-corrected AI scores (subtract per-dimension mean bias)")
    md.p("Removes systematic AI over-rating per dimension before computing error metrics. "
         "Correlation is unchanged; MAE/RMSE improve if the main source of error is a constant offset.")
    md.table(_score_alignment_table(merged_scores_bc, DIMENSIONS))

    if multi_run:
        md.h3(f"F. Human mean vs AI mean-of-{n_ai_runs}-runs")
        md.table(_score_alignment_table(merged_scores_ai_mean, DIMENSIONS))

        md.h3(f"G. Human mean vs AI median-of-{n_ai_runs}-runs")
        md.table(_score_alignment_table(merged_scores_ai_median, DIMENSIONS))

        md.h3(f"H. Human median vs AI median-of-{n_ai_runs}-runs")
        md.table(_score_alignment_table(merged_scores_hmed_aimed, DIMENSIONS))

        md.h3(f"I. 3-annotator episodes, Human mean vs AI mean-of-{n_ai_runs}-runs")
        md.p(f"Episodes: {merged_scores_3ann_ai_mean['episode_id'].nunique()}")
        md.table(_score_alignment_table(merged_scores_3ann_ai_mean, DIMENSIONS))

        md.h3(f"J. 3-annotator episodes, Human mean vs AI median-of-{n_ai_runs}-runs")
        md.p(f"Episodes: {merged_scores_3ann_ai_median['episode_id'].nunique()}")
        md.table(_score_alignment_table(merged_scores_3ann_ai_median, DIMENSIONS))

    # Per-annotator
    md.h3("Per-annotator vs AI score correlation (Spearman ρ)")
    annotators = sorted(human_scores["annotator"].unique())
    rows = []
    for ann in annotators:
        ann_scores = human_scores[human_scores["annotator"] == ann]
        merged_ann = ann_scores.merge(
            ai_scores[["episode_id", "dimension", "agent_idx", "ai_score"]],
            on=["episode_id", "dimension", "agent_idx"], how="inner"
        )
        if len(merged_ann) < 5:
            continue
        row = {"Annotator": ann, "N": len(merged_ann)}
        h, a = merged_ann["score"].values, merged_ann["ai_score"].values
        row["Overall ρ"] = fmt(_safe_spearman(h, a), 3)
        for dim in DIMENSIONS:
            sub = merged_ann[merged_ann["dimension"] == dim]
            if len(sub) < 3 or sub["score"].std() == 0 or sub["ai_score"].std() == 0:
                row[dim] = "N/A"
            else:
                row[dim] = fmt(stats.spearmanr(sub["score"], sub["ai_score"])[0], 3)
        rows.append(row)
    md.table(rows)

    # =====================================================================
    # ATTRIBUTIONS
    # =====================================================================
    md.h2("18. Human vs AI — Attribution Alignment")

    md.h3("A. Human mean, all episodes (baseline)")
    md.table(_attr_alignment_table(merged_attr, DIMENSIONS))

    md.h3("B. Human median, all episodes")
    md.table(_attr_alignment_table(merged_attr_median, DIMENSIONS))

    md.h3("C. Human mean, 3-annotator episodes only")
    md.p(f"Episodes: {merged_attr_3ann['episode_id'].nunique()}")
    md.table(_attr_alignment_table(merged_attr_3ann, DIMENSIONS))

    md.h3("D. Human median, 3-annotator episodes only")
    md.p(f"Episodes: {merged_attr_3ann_median['episode_id'].nunique()}")
    md.table(_attr_alignment_table(merged_attr_3ann_median, DIMENSIONS))

    md.h3("E. Within-episode rank correlation (per-episode Spearman, then averaged)")
    md.p("Tests whether AI preserves the *relative ordering* of turn importance within each episode, "
         "removing cross-episode calibration differences.")
    md.table(_within_episode_rank_corr(merged_attr, DIMENSIONS))

    md.h3("F. Within-episode rank correlation — 3-annotator episodes only")
    md.table(_within_episode_rank_corr(merged_attr_3ann, DIMENSIONS))

    md.h3("G. Per-episode rank-normalized attribution correlation")
    md.p("Rank-normalizes attributions within each (episode, dimension) before computing global correlation. "
         "This removes the effect of AI's systematically higher calibration while preserving relative ordering.")
    md.table(_rank_normalized_attr_table(merged_attr, DIMENSIONS))

    md.h3("H. Per-episode rank-normalized — 3-annotator episodes only")
    md.table(_rank_normalized_attr_table(merged_attr_3ann, DIMENSIONS))

    if multi_run:
        md.h3(f"I. Human mean vs AI mean-of-{n_ai_runs}-runs")
        md.table(_attr_alignment_table(merged_attr_ai_mean, DIMENSIONS))

        md.h3(f"J. Human mean vs AI median-of-{n_ai_runs}-runs")
        md.table(_attr_alignment_table(merged_attr_ai_median, DIMENSIONS))

        md.h3(f"K. Human median vs AI median-of-{n_ai_runs}-runs")
        md.table(_attr_alignment_table(merged_attr_hmed_aimed, DIMENSIONS))

        md.h3(f"L. 3-annotator, Human mean vs AI mean-of-{n_ai_runs}-runs")
        md.p(f"Episodes: {merged_attr_3ann_ai_mean['episode_id'].nunique()}")
        md.table(_attr_alignment_table(merged_attr_3ann_ai_mean, DIMENSIONS))

        md.h3(f"M. 3-annotator, Human mean vs AI median-of-{n_ai_runs}-runs")
        md.p(f"Episodes: {merged_attr_3ann_ai_median['episode_id'].nunique()}")
        md.table(_attr_alignment_table(merged_attr_3ann_ai_median, DIMENSIONS))

        md.h3(f"N. Rank-normalized, AI mean-of-{n_ai_runs}-runs")
        md.table(_rank_normalized_attr_table(merged_attr_ai_mean, DIMENSIONS))

        md.h3(f"O. Rank-normalized, 3-annotator, AI mean-of-{n_ai_runs}-runs")
        md.table(_rank_normalized_attr_table(merged_attr_3ann_ai_mean, DIMENSIONS))

    # Confusion matrix
    md.h3("Attribution score confusion (human rounded vs AI, 0-3 scale)")
    h_rounded = np.clip(np.round(merged_attr["human_attr"].values).astype(int), 0, 3)
    a_vals = np.clip(merged_attr["ai_attribution"].values.astype(int), 0, 3)
    confusion = pd.crosstab(pd.Series(h_rounded, name="Human"), pd.Series(a_vals, name="AI"), margins=True)
    md.table_df(confusion.reset_index())

    # Per-annotator
    md.h3("Per-annotator vs AI attribution correlation (Spearman ρ)")
    rows = []
    for ann in annotators:
        ann_a = human_attr[human_attr["annotator"] == ann].rename(columns={"attribution": "human_attr"})
        merged_ann = ann_a.merge(
            ai_attr[["episode_id", "dimension", "turn_key", "ai_attribution"]],
            on=["episode_id", "dimension", "turn_key"], how="inner"
        )
        if len(merged_ann) < 5:
            continue
        row = {"Annotator": ann, "N": len(merged_ann)}
        h, a = merged_ann["human_attr"].values, merged_ann["ai_attribution"].values
        row["Overall ρ"] = fmt(_safe_spearman(h, a), 3)
        for dim in DIMENSIONS:
            sub = merged_ann[merged_ann["dimension"] == dim]
            if len(sub) < 3 or sub["human_attr"].std() == 0 or sub["ai_attribution"].std() == 0:
                row[dim] = "N/A"
            else:
                row[dim] = fmt(stats.spearmanr(sub["human_attr"], sub["ai_attribution"])[0], 3)
        rows.append(row)
    md.table(rows)

    # =====================================================================
    # TURN POSITION
    # =====================================================================
    md.h2("19. Human vs AI — Turn Position Attribution Patterns")
    merged_attr_cp = merged_attr.copy()
    merged_attr_cp["turn_num"] = merged_attr_cp["turn_key"].str.extract(r"turn_(\d+)").astype(int)
    turn_corr = []
    for turn_num in sorted(merged_attr_cp["turn_num"].unique()):
        sub = merged_attr_cp[merged_attr_cp["turn_num"] == turn_num]
        if len(sub) < 5:
            continue
        h, a = sub["human_attr"].values, sub["ai_attribution"].values
        turn_corr.append({
            "Turn": turn_num, "N": len(sub),
            "Spearman ρ": fmt(_safe_spearman(h, a), 3),
            "Human mean": fmt(np.mean(h), 2), "AI mean": fmt(np.mean(a), 2),
            "MAE": fmt(mean_absolute_error(h, a), 2),
        })
    md.table(turn_corr)

    # =====================================================================
    # RANK AGREEMENT
    # =====================================================================
    md.h2("20. Human vs AI — Score Rank Agreement")
    md.p("For each episode, do human and AI agree on which agent scored higher?")
    rank_rows = []
    for dim in DIMENSIONS:
        sub = merged_scores[merged_scores["dimension"] == dim]
        agree, total = 0, 0
        for eid in sub["episode_id"].unique():
            ep = sub[sub["episode_id"] == eid]
            if len(ep) != 2:
                continue
            h1, h2 = ep[ep["agent_idx"] == 1]["human_score"].values[0], ep[ep["agent_idx"] == 2]["human_score"].values[0]
            a1, a2 = ep[ep["agent_idx"] == 1]["ai_score"].values[0], ep[ep["agent_idx"] == 2]["ai_score"].values[0]
            h_w = 1 if h1 > h2 else (2 if h2 > h1 else 0)
            a_w = 1 if a1 > a2 else (2 if a2 > a1 else 0)
            if h_w == a_w:
                agree += 1
            total += 1
        rank_rows.append({"Dimension": dim, "N ep": total,
                          "Rank agree %": f"{agree / total * 100:.1f}%" if total > 0 else "N/A"})
    md.table(rank_rows)

    # =====================================================================
    # DIVERGENCE
    # =====================================================================
    md.h2("21. Human vs AI — Per-Episode Score Differences")
    md.p("Episodes where human and AI scores diverge the most.")
    merged_scores["diff"] = (merged_scores["ai_score"] - merged_scores["human_score"]).abs()
    ep_diffs = merged_scores.groupby("episode_id")["diff"].mean().sort_values(ascending=False)
    md.table([{"Episode": eid, "Mean |AI-Human|": fmt(v, 2)} for eid, v in ep_diffs.head(10).items()])

    md.h3("Dimensions where AI diverges most from humans")
    dim_diffs = merged_scores.groupby("dimension")["diff"].mean().sort_values(ascending=False)
    md.table([{"Dimension": d, "Mean |AI-Human|": fmt(v, 2)} for d, v in dim_diffs.items()])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _build_report(
    md: MarkdownWriter,
    scores_df: pd.DataFrame, attr_df: pd.DataFrame,
    episode_models: dict, ep_counts: pd.DataFrame,
    combined_only: bool = False,
):
    """Build sections 0-10 of the report. If combined_only, skip per-split subsections."""
    splits = sorted(scores_df["split"].unique())

    scores_2 = filter_by_annotator_count(scores_df, ep_counts, 2)
    attr_2 = filter_by_annotator_count(attr_df, ep_counts, 2)
    scores_3 = filter_by_annotator_count(scores_df, ep_counts, 3)
    attr_3 = filter_by_annotator_count(attr_df, ep_counts, 3)

    def _split_iter(df, col="split"):
        """Yield (label, subset) pairs: per-split if not combined_only, always combined."""
        if not combined_only:
            for s in splits:
                yield s, df[df[col] == s]
        yield "Combined", df

    # === Section 0 ===
    md.h2("0. Per-Episode Annotation Counts")

    if not combined_only:
        for split in splits:
            md.h3(f"Split: {split}")
            sub = ep_counts[ep_counts["split"] == split].sort_values("episode_id")
            count_summary = sub["n_annotators"].value_counts().sort_index()
            md.p(f"**Total episodes:** {len(sub)}")
            for n, cnt in count_summary.items():
                md.bullet(f"{cnt} episodes with {n} annotator(s)")
            md.end_bullets()
            rows_table = []
            for _, row in sub.iterrows():
                ann_for_ep = sorted(
                    scores_df[(scores_df["split"] == split) & (scores_df["episode_id"] == row["episode_id"])]["annotator"].unique()
                )
                rows_table.append({"Episode ID": row["episode_id"], "# Annotators": int(row["n_annotators"]),
                                   "Annotators": ", ".join(ann_for_ep)})
            md.table(rows_table)

    sub_all = ep_counts.sort_values("episode_id")
    count_summary = sub_all["n_annotators"].value_counts().sort_index()
    md.h3("Combined" if combined_only else "Combined (all splits)")
    md.p(f"**Total episodes:** {len(sub_all)}")
    for n, cnt in count_summary.items():
        md.bullet(f"{cnt} episodes with {n} annotator(s)")
    md.end_bullets()

    md.h3("Episode counts by annotator threshold")
    for split in splits:
        n2 = len(ep_counts[(ep_counts["split"] == split) & (ep_counts["n_annotators"] >= 2)])
        n3 = len(ep_counts[(ep_counts["split"] == split) & (ep_counts["n_annotators"] >= 3)])
        md.bullet(f"**{split}:** {n2} episodes with ≥2 annotators, {n3} with 3 annotators")
    n2_all = len(ep_counts[ep_counts["n_annotators"] >= 2])
    n3_all = len(ep_counts[ep_counts["n_annotators"] >= 3])
    md.bullet(f"**Combined:** {n2_all} episodes with ≥2 annotators, {n3_all} with 3 annotators")
    md.end_bullets()

    # === Sections 1–2: Agreement ===
    for sec_num, sec_title, pivot_fn, value_col, show_kappa in [
        (1, "Dimension Score Agreement", pivot_scores, "score", False),
        (2, "Attribution Score Agreement", pivot_attributions, "attribution", True),
    ]:
        md.h2(f"{sec_num}. {sec_title}")
        for min_ann, label in [(2, "Pairwise (≥2 annotators)"), (3, "3-annotator only")]:
            data = filter_by_annotator_count(
                scores_df if value_col == "score" else attr_df, ep_counts, min_ann
            )
            for lbl, sub in _split_iter(data):
                if sub.empty:
                    md.h3(f"{lbl}, {label} — SKIPPED")
                    continue
                md.h3(f"{lbl}, {label} — Overall")
                piv = pivot_fn(sub)
                report_agreement(md, piv)
                md.h3(f"{lbl}, {label} — Per dimension")
                md.table(agreement_per_dim_table(pivot_fn, sub, show_kappa))

    # === Section 2b: Binned Attribution Agreement (0, 1-2, 3) ===
    md.h2("2b. Attribution Agreement — Binned (0 / 1-2 / 3)")
    md.p("Attribution scores binned into 3 levels: **0** (no impact), **1-2** (minor/significant), **3** (critical). "
         "This reduces noise from fine-grained disagreements between 1 and 2.")
    for min_ann, label in [(2, "Pairwise (≥2 annotators)"), (3, "3-annotator only")]:
        data = filter_by_annotator_count(attr_df, ep_counts, min_ann)
        for lbl, sub in _split_iter(data):
            if sub.empty:
                md.h3(f"{lbl}, {label} — SKIPPED")
                continue
            md.h3(f"{lbl}, {label} — Overall (binned)")
            piv = pivot_attributions_binned(sub)
            report_agreement(md, piv)
            md.h3(f"{lbl}, {label} — Per dimension (binned)")
            md.table(agreement_per_dim_table(pivot_attributions_binned, sub, show_kappa=True))

    # === Sections 3–4: Total agreement ===
    for sec_num, sec_title, source_col, group_idx in [
        (3, "Total Score Agreement (sum across 7 dimensions)", "score", ["episode_id", "agent_idx"]),
        (4, "Total Attribution Agreement (sum across 7 dimensions)", "attribution", ["episode_id", "turn_key"]),
    ]:
        md.h2(f"{sec_num}. {sec_title}")
        for min_ann, label in [(2, "Pairwise (≥2 annotators)"), (3, "3-annotator only")]:
            sub_data = (scores_2 if source_col == "score" else attr_2) if min_ann == 2 else (scores_3 if source_col == "score" else attr_3)
            totals = sub_data.groupby(["split", "annotator"] + group_idx)[source_col].sum().reset_index()
            for lbl, sub in _split_iter(totals):
                if sub.empty:
                    md.h3(f"{lbl}, {label} — SKIPPED")
                    continue
                md.h3(f"{lbl}, {label}")
                piv = sub.pivot_table(index=group_idx, columns="annotator", values=source_col)
                report_agreement(md, piv)

    # === Section 5: Descriptive stats — scores ===
    md.h2("5. Descriptive Statistics — Dimension Scores")
    for label, sub in _split_iter(scores_df):
        md.h3(f"{label} — Per-dimension mean ± std")
        rows = []
        for dim in DIMENSIONS:
            d = sub[sub["dimension"] == dim]["score"].dropna()
            if d.empty:
                continue
            rows.append({"Dimension": dim, "Mean": fmt(d.mean(), 2), "Std": fmt(d.std(), 2),
                         "Median": fmt(d.median(), 1), "Min": f"{d.min():.0f}", "Max": f"{d.max():.0f}", "N": len(d)})
        md.table(rows)

        md.h3(f"{label} — Per-annotator mean score (annotator bias)")
        bias = []
        for ann in sorted(sub["annotator"].unique()):
            ad = sub[sub["annotator"] == ann]
            row = {"Annotator": ann}
            for dim in DIMENSIONS:
                v = ad[ad["dimension"] == dim]["score"].dropna()
                row[dim] = fmt(v.mean(), 2) if len(v) > 0 else "N/A"
            row["Overall"] = fmt(ad["score"].dropna().mean(), 2)
            bias.append(row)
        md.table(bias)

    # === Section 6: Descriptive stats — attributions ===
    md.h2("6. Descriptive Statistics — Attributions")
    for label, sub in _split_iter(attr_df):
        md.h3(f"{label} — Per-dimension attribution mean ± std")
        rows = []
        for dim in DIMENSIONS:
            d = sub[sub["dimension"] == dim]["attribution"].dropna()
            if d.empty:
                continue
            nz = f"{(d != 0).mean() * 100:.1f}%" if len(d) > 0 else "0%"
            rows.append({"Dimension": dim, "Mean": fmt(d.mean(), 2), "Std": fmt(d.std(), 2),
                         "% Non-zero": nz, "N": len(d)})
        md.table(rows)

    # === Section 7: Correlation matrix ===
    md.h2("7. Dimension Score Correlation Matrix (Spearman)")
    for label, sub in _split_iter(scores_df):
        md.h3(f"{label}")
        wide = sub.pivot_table(index=["episode_id", "annotator", "agent_idx"], columns="dimension", values="score")
        if wide.shape[0] < 5:
            md.p("*Too few data points.*")
            continue
        corr = wide[DIMENSIONS].corr(method="spearman").round(3)
        md.table_df(corr.reset_index().rename(columns={"dimension": ""}))

    # === Section 8: Disagreement ===
    md.h2("8. Per-Episode Disagreement Analysis")
    for min_ann, label, s_data in [(2, "Pairwise (≥2)", scores_2), (3, "3-annotator", scores_3)]:
        for lbl, sub in _split_iter(s_data):
            if sub.empty:
                md.h3(f"{lbl}, {label} — SKIPPED")
                continue
            md.h3(f"{lbl}, {label} — Top disagreement episodes")
            piv = pivot_scores(sub)
            if piv.empty:
                continue
            piv["range"] = piv.max(axis=1) - piv.min(axis=1)
            ep_dis = piv.reset_index().groupby("episode_id")["range"].mean().sort_values(ascending=False)
            md.table([{"Episode": eid, "Mean score range": fmt(v, 2)} for eid, v in ep_dis.head(5).items()])
            md.h3(f"{lbl}, {label} — Dimensions with highest disagreement")
            dim_dis = piv.reset_index().groupby("dimension")["range"].mean().sort_values(ascending=False)
            md.table([{"Dimension": d, "Mean score range": fmt(v, 2)} for d, v in dim_dis.items()])

    # === Section 9: Zero-variance ===
    md.h2("9. Zero-Variance / Low-Variance Dimension Analysis")
    for label, sub in _split_iter(scores_df):
        md.h3(f"{label}")
        rows = []
        for dim in DIMENSIONS:
            d = sub[sub["dimension"] == dim]["score"].dropna()
            if d.empty:
                continue
            rows.append({"Dimension": dim, "Unique values": d.nunique(),
                         "Variance": fmt(d.var(), 2), "% Zero scores": f"{(d == 0).mean() * 100:.1f}%"})
        md.table(rows)

    # === Section 10: Turn position ===
    md.h2("10. Attribution Patterns — Turn Position Analysis")
    for label, sub_raw in _split_iter(attr_df):
        md.h3(f"{label} — Mean attribution by turn position")
        sub = sub_raw.copy()
        sub["turn_num"] = sub["turn_key"].str.extract(r"turn_(\d+)").astype(int)
        sub["agent"] = sub["turn_key"].str.extract(r"(agent_\d)")
        tm = sub.groupby("turn_num")["attribution"].agg(["mean", "std", "count"]).reset_index()
        md.table([{"Turn": int(r["turn_num"]), "Mean attr": fmt(r["mean"], 3),
                   "Std": fmt(r["std"], 3), "N": int(r["count"])} for _, r in tm.iterrows()])
        md.h3(f"{label} — Agent 1 vs Agent 2 mean attribution")
        for agent, val in sub.groupby("agent")["attribution"].mean().items():
            md.bullet(f"**{agent}:** {fmt(val, 3)}")
        md.end_bullets()


def main():
    episode_models = load_episode_models()
    scores_df = load_all_annotations(episode_models)
    attr_df = load_all_attributions(episode_models)

    splits = sorted(scores_df["split"].unique())
    annotators = sorted(scores_df["annotator"].unique())
    ep_counts = get_episode_annotator_counts(scores_df)
    ai_scores, ai_attr = load_ai_annotations()

    def _write_report(path: Path, title: str, combined_only: bool):
        md = MarkdownWriter()
        md.h1(title)
        md.p(f"**Score records:** {len(scores_df)} | **Attribution records:** {len(attr_df)}")
        md.p(f"**Splits:** {', '.join(splits)} | **Annotators:** {', '.join(annotators)}")
        if episode_models:
            md.p(f"**Model info:** loaded for {len(episode_models)} episodes")
        else:
            md.p("*No episode_models.json found — run `fetch_episode_models.py` first.*")

        _build_report(md, scores_df, attr_df, episode_models, ep_counts, combined_only=combined_only)

        # Model sections (11-15) only in the full report
        if not combined_only:
            md.h2("11. Episode Model Mapping")
            if not episode_models:
                md.p("*Skipped — no episode_models.json. Run `fetch_episode_models.py` first.*")
            else:
                for split in splits:
                    md.h3(f"{split}")
                    sub = scores_df[scores_df["split"] == split]
                    ep_ids = sorted(sub["episode_id"].unique())
                    rows = []
                    for eid in ep_ids:
                        info = episode_models.get(eid, {})
                        names = sub[sub["episode_id"] == eid][["agent_idx", "agent_name"]].drop_duplicates()
                        a1 = names[names["agent_idx"] == 1]["agent_name"].iloc[0] if len(names[names["agent_idx"] == 1]) > 0 else ""
                        a2 = names[names["agent_idx"] == 2]["agent_name"].iloc[0] if len(names[names["agent_idx"] == 2]) > 0 else ""
                        rows.append({"Episode ID": eid, "Agent 1": a1, "Agent 1 Model": info.get("agent_1_model", "N/A"),
                                     "Agent 2": a2, "Agent 2 Model": info.get("agent_2_model", "N/A")})
                    md.table(rows)

            has_models = scores_df["model"].notna() & (scores_df["model"] != "")
            if has_models.any():
                models = sorted(scores_df.loc[has_models, "model"].unique())

                md.h2("12. Per-Model Score Analysis")
                md.p(f"**Models:** {', '.join(models)}")
                for split in splits:
                    md.h3(f"{split} — Mean score per model per dimension")
                    sub = scores_df[(scores_df["split"] == split) & has_models]
                    if sub.empty:
                        continue
                    rows = []
                    for model in models:
                        m = sub[sub["model"] == model]
                        if m.empty:
                            continue
                        row = {"Model": model, "N ep": m["episode_id"].nunique()}
                        for dim in DIMENSIONS:
                            v = m[m["dimension"] == dim]["score"].dropna()
                            row[dim] = fmt(v.mean(), 2) if len(v) > 0 else "N/A"
                        row["Overall"] = fmt(m["score"].dropna().mean(), 2)
                        rows.append(row)
                    md.table(rows)

                md.h3("All splits — Per-model mean score")
                rows = []
                for model in models:
                    m = scores_df[scores_df["model"] == model]
                    if m.empty:
                        continue
                    row = {"Model": model, "N ep": m["episode_id"].nunique()}
                    for dim in DIMENSIONS:
                        v = m[m["dimension"] == dim]["score"].dropna()
                        row[dim] = fmt(v.mean(), 2) if len(v) > 0 else "N/A"
                    row["Overall"] = fmt(m["score"].dropna().mean(), 2)
                    rows.append(row)
                md.table(rows)

                md.h2("13. Per-Model Attribution Analysis")
                attr_has_models = attr_df["model"].notna() & (attr_df["model"] != "")
                for split in splits:
                    md.h3(f"{split} — Mean attribution per model per dimension")
                    sub = attr_df[(attr_df["split"] == split) & attr_has_models]
                    if sub.empty:
                        continue
                    rows = []
                    for model in models:
                        m = sub[sub["model"] == model]
                        if m.empty:
                            continue
                        row = {"Model": model}
                        for dim in DIMENSIONS:
                            v = m[m["dimension"] == dim]["attribution"].dropna()
                            row[dim] = fmt(v.mean(), 2) if len(v) > 0 else "N/A"
                        row["Overall"] = fmt(m["attribution"].dropna().mean(), 2)
                        rows.append(row)
                    md.table(rows)

                md.h2("14. Per-Model Scores — Position-Agnostic")
                def model_table_with_position(sub_df):
                    rows = []
                    for model in models:
                        m = sub_df[sub_df["model"] == model]
                        if m.empty:
                            continue
                        row = {"Model": model,
                               "As a1": m[m["agent_idx"] == 1]["episode_id"].nunique(),
                               "As a2": m[m["agent_idx"] == 2]["episode_id"].nunique()}
                        for dim in DIMENSIONS:
                            v = m[m["dimension"] == dim]["score"].dropna()
                            row[dim] = fmt(v.mean(), 2) if len(v) > 0 else "N/A"
                        row["Overall"] = fmt(m["score"].dropna().mean(), 2)
                        rows.append(row)
                    return rows

                for split in splits:
                    sub = scores_df[(scores_df["split"] == split) & has_models]
                    if sub.empty:
                        continue
                    md.h3(f"{split} — All episodes (including self-play)")
                    md.table(model_table_with_position(sub))
                    ep_pairs = (sub[["episode_id", "agent_idx", "model"]].drop_duplicates()
                                .pivot(index="episode_id", columns="agent_idx", values="model")
                                .rename(columns={1: "model_1", 2: "model_2"}))
                    cross_eps = ep_pairs[ep_pairs["model_1"] != ep_pairs["model_2"]].index
                    sub_cross = sub[sub["episode_id"].isin(cross_eps)]
                    if not sub_cross.empty:
                        md.h3(f"{split} — Cross-play only (excluding self-play)")
                        md.table(model_table_with_position(sub_cross))

                md.h3("All splits — All episodes")
                md.table(model_table_with_position(scores_df[has_models]))
                all_pairs = (scores_df[has_models][["split", "episode_id", "agent_idx", "model"]].drop_duplicates()
                             .pivot(index=["split", "episode_id"], columns="agent_idx", values="model")
                             .rename(columns={1: "model_1", 2: "model_2"}))
                cross_all = all_pairs[all_pairs["model_1"] != all_pairs["model_2"]].index
                cross_eids = {eid for _, eid in cross_all}
                sub_cross_all = scores_df[scores_df["episode_id"].isin(cross_eids) & has_models]
                if not sub_cross_all.empty:
                    md.h3("All splits — Cross-play only")
                    md.table(model_table_with_position(sub_cross_all))

                md.h2("15. Model Pair Comparison (head-to-head)")
                for split in splits:
                    md.h3(f"{split}")
                    sub = scores_df[(scores_df["split"] == split) & has_models]
                    if sub.empty:
                        continue
                    ep_pairs = (sub[["episode_id", "agent_idx", "model"]].drop_duplicates()
                                .pivot(index="episode_id", columns="agent_idx", values="model")
                                .rename(columns={1: "model_1", 2: "model_2"}))
                    pair_counts = ep_pairs.groupby(["model_1", "model_2"]).size().reset_index(name="n_episodes")
                    md.table(pair_counts.to_dict("records"))
                    for _, pr in pair_counts.iterrows():
                        m1, m2 = pr["model_1"], pr["model_2"]
                        p_eps = ep_pairs[(ep_pairs["model_1"] == m1) & (ep_pairs["model_2"] == m2)].index
                        ps = sub[sub["episode_id"].isin(p_eps)]
                        col1 = f"{m1} (a1)" if m1 == m2 else m1
                        col2 = f"{m2} (a2)" if m1 == m2 else m2
                        md.h3(f"{m1} (agent_1) vs {m2} (agent_2) — {len(p_eps)} episodes")
                        rows = []
                        for dim in DIMENSIONS:
                            dd = ps[ps["dimension"] == dim]
                            s1 = dd[dd["agent_idx"] == 1]["score"].dropna()
                            s2 = dd[dd["agent_idx"] == 2]["score"].dropna()
                            diff = fmt(s1.mean() - s2.mean(), 2) if len(s1) > 0 and len(s2) > 0 else "N/A"
                            rows.append({"Dimension": dim, f"{col1} mean": fmt(s1.mean(), 2) if len(s1) else "N/A",
                                         f"{col2} mean": fmt(s2.mean(), 2) if len(s2) else "N/A", "Diff (a1-a2)": diff})
                        t1 = ps[ps["agent_idx"] == 1]["score"].dropna()
                        t2 = ps[ps["agent_idx"] == 2]["score"].dropna()
                        rows.append({"Dimension": "**OVERALL**", f"{col1} mean": fmt(t1.mean(), 2),
                                     f"{col2} mean": fmt(t2.mean(), 2), "Diff (a1-a2)": fmt(t1.mean() - t2.mean(), 2)})
                        md.table(rows)

        # Human vs AI alignment (in both reports)
        if not ai_scores.empty:
            human_ai_alignment_analysis(md, scores_df, attr_df, ai_scores, ai_attr)

        report = md.getvalue()
        with open(path, "w") as f:
            f.write(report)
        print(f"Report written to {path} ({len(report)} chars)")

    _write_report(REPORT_PATH, "Human Evaluation Annotation Analysis", combined_only=False)
    _write_report(REPORT_COMBINED_PATH, "Human Evaluation Annotation Analysis (Combined)", combined_only=True)


if __name__ == "__main__":
    main()
