"""W&B shared-run codebook charts: layer mean with a shaded SEM band."""

import math


METRICS = ("utilization", "perplexity", "dominant_fraction")
PRESET_NAMES = {"linear": "cw-codebook-mean-sem-linear-v1",
                "log": "cw-codebook-mean-sem-log-v1"}


def mean_sem_spec(log_scale=False):
    """A reusable Vega-Lite preset; runs share axes but never share statistics."""
    scale = {"type": "log" if log_scale else "linear", "zero": False}
    x = {"field": "${field:step}", "type": "quantitative", "title": "Training step"}
    y = {"field": "${field:mean}", "type": "quantitative",
         "title": "${string:metric}", "scale": scale, "stack": None}
    layers = [
        {"transform": [{"filter": "isValid(datum['${field:lower}']) && isValid(datum['${field:upper}'])"}],
         "mark": {"type": "area", "opacity": 0.15},
         "encoding": {"x": x, "y": {**y, "field": "${field:lower}"},
                      "y2": {"field": "${field:upper}"},
                      "detail": {"field": "${field:band_segment}", "type": "nominal"}}},
        {"mark": {"type": "line", "point": True},
         "encoding": {"x": x, "y": y,
                      "detail": {"field": "${field:mean_segment}", "type": "nominal"},
                      "tooltip": [{"field": "${field:run}", "type": "nominal", "title": "Run"},
                                  {**x, "format": ".0f"},
                                  {"field": "${field:mean}", "type": "quantitative", "title": "Layer mean"},
                                  {"field": "${field:sem}", "type": "quantitative", "title": "Layer SEM"}]}}
    ]
    if log_scale:
        layers.append({
            "transform": [{"aggregate": [{"op": "max", "field": "${field:uniform}", "as": "uniform"}],
                           "groupby": ["${field:run}"]}],
            "mark": {"type": "rule", "strokeDash": [6, 4], "opacity": 0.6},
            "encoding": {"y": {**y, "field": "uniform"},
                         "tooltip": [{"field": "uniform", "type": "quantitative", "title": "Uniform: 1/N"}]},
        })
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"name": "wandb"}, "title": "${string:title}",
        "encoding": {"color": {"field": "${field:run}", "type": "nominal", "title": "Run"}},
        "layer": layers,
        "resolve": {"scale": {"x": "shared", "y": "shared", "color": "shared"}},
    }


def summary_rows(history, metric, run_name, run_id):
    """Preserve per-run SEM; an unobserved/single layer never gets fake error bars."""
    rows = []
    band_segment = mean_segment = 0
    uniform = 1 / history["shape"][1]
    for row in history["rows"]:
        mean, sem = row[f"{metric}_mean"], row[f"{metric}_sem"]
        mean = mean if mean is not None and math.isfinite(mean) else None
        sem = sem if sem is not None and math.isfinite(sem) else None
        lower = mean - sem if mean is not None and sem is not None else None
        upper = mean + sem if lower is not None else None
        if metric == "dominant_fraction":
            # The theoretical lower bound is 1/N; guard against roundoff too.
            if mean is not None and mean <= 0:
                mean, lower, upper = None, None, None
            if lower is not None:
                lower = max(uniform, lower)
        # Separate paths on either side of missing data, even if Vega filters
        # invalid rows. A single observed layer must not acquire an implied band.
        band_segment += lower is None
        mean_segment += mean is None
        rows.append([f"{run_name} ({run_id})", row["step"], mean, sem, lower, upper, uniform,
                     band_segment, mean_segment])
    return rows


def wandb_summary_charts(history, split, wandb, run, presets):
    """Native custom charts query the tables of all selected runs in W&B."""
    columns = ["run", "step", "mean", "sem", "lower", "upper", "uniform",
               "band_segment", "mean_segment"]
    result = {}
    for metric in METRICS:
        table = wandb.Table(columns=columns, data=summary_rows(history, metric, run.name, run.id))
        result[f"{split}_codebooks/{metric}"] = wandb.plot_table(
            presets["log" if metric == "dominant_fraction" else "linear"], table,
            fields={name: name for name in columns},
            string_fields={"title": f"{split.capitalize()} {metric}: layer mean ± SEM",
                           "metric": metric.replace("_", " ")},
            # Keep required backing tables outside the two codebook plot sections.
            split_table=True,
        )
    return result
