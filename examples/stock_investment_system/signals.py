"""Generate today's trading signal from a model trained via qlib's workflow.

Prerequisites: the training/backtest workflow has already been run once with
`qrun workflow_config_us_lightgbm.yaml` (see README.md), which logs a trained
model ("params.pkl") and its dataset config ("dataset") to a qlib/mlflow
experiment. This module reloads the most recently *finished* recorder from
that experiment and re-runs inference for a single, fresh trading day instead
of the fixed backtest date range the config specifies.

Reuses qlib.workflow.online.update.RMDLoader, the same helper qlib's own
"online serving" examples (examples/online_srv) use to refresh predictions,
so the inference path here matches an already-reviewed part of qlib.
"""

from __future__ import annotations

import pandas as pd
import qlib
from qlib.data import D
from qlib.workflow import R
from qlib.workflow.online.update import RMDLoader
from qlib.workflow.recorder import Recorder


class NoTrainedModelError(RuntimeError):
    """Raised when the experiment has no finished training run to load."""


def get_latest_recorder(experiment_name: str) -> Recorder:
    exp = R.get_exp(experiment_name=experiment_name, create=False)
    recorders = exp.list_recorders(rtype="list")
    finished = [r for r in recorders if r.status == Recorder.STATUS_FI]
    if not finished:
        raise NoTrainedModelError(
            f"No finished recorder found in experiment {experiment_name!r}. "
            f"Run `qrun workflow_config_us_lightgbm.yaml` first to train a model."
        )
    return max(finished, key=lambda r: r.start_time)


def generate_today_signal(
    experiment_name: str,
    provider_uri: str,
    region: str,
    predict_date: str | None = None,
    lookback_days: int = 260,
) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
    """Return (scores, prices, predict_date) for the requested trading day.

    scores: predicted signal per instrument, sorted descending (best first).
    prices: that day's close price per instrument, used to size orders.
    predict_date: the actual trading day used (defaults to the latest one
        available in the qlib calendar).
    """
    qlib.init(provider_uri=provider_uri, region=region)

    recorder = get_latest_recorder(experiment_name)
    loader = RMDLoader(rec=recorder)
    model = loader.get_model()

    calendar = D.calendar(freq="day")
    if not len(calendar):
        raise RuntimeError("qlib calendar is empty -- has market data been downloaded for this provider_uri?")

    if predict_date is None:
        predict_date = calendar[-1]
    else:
        predict_date = pd.Timestamp(predict_date)
        if predict_date not in calendar:
            raise ValueError(f"{predict_date} is not a trading day in the qlib calendar for region={region!r}")

    predict_idx = calendar.get_loc(predict_date) if hasattr(calendar, "get_loc") else list(calendar).index(predict_date)
    start_idx = max(0, predict_idx - lookback_days)
    start_buffer = calendar[start_idx]

    dataset = loader.get_dataset(
        start_time=start_buffer,
        end_time=predict_date,
        segments={"test": (predict_date, predict_date)},
    )

    pred = model.predict(dataset)
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    if "datetime" in (pred.index.names or []):
        pred = pred.groupby(level="instrument").last()
    scores = pred.sort_values(ascending=False)
    scores = scores[~scores.index.duplicated(keep="last")]

    symbols = list(scores.index)
    if not symbols:
        raise RuntimeError(f"Model produced no predictions for {predict_date}. Check the instrument universe/data.")

    close_df = D.features(symbols, ["$close"], start_time=predict_date, end_time=predict_date, freq="day")
    prices = close_df["$close"].groupby(level="instrument").last()

    return scores, prices, predict_date
