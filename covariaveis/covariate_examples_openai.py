"""
Examples of zero‑shot time–series forecasting with dynamic covariates for
a variety of foundation models.  These examples are inspired by usage
guidelines and code snippets from the official model cards, blogs and
repositories.  Each example illustrates how to organise the target
series and dynamic covariate series, load the corresponding pre‑trained
model and obtain a forecast.

The examples are meant to serve as templates.  They do not perform
training or evaluation and may need to be adapted to suit your data
format.  See the accompanying report for citations and background.
"""

import numpy as np
import pandas as pd


def example_chronos2_with_covariates():
    """Forecast using Amazon Chronos‑2 with dynamic covariates.

    Chronos‑2 accepts past and future real/categorical covariates when
    calling the `predict_df` method.  The `context_df` contains the
    historical target and past covariate values.  The optional
    `future_df` holds future covariate values over the forecast
    horizon.  Column names are passed via parameters.  See the
    Chronos‑2 model card for details【661168599961157†L96-L133】.
    """
    try:
        from chronos_forecasting import Chronos2Pipeline
    except ImportError:
        raise RuntimeError("chronos-forecasting library is not installed")

    # Example synthetic dataset with a target and two covariates
    dates = pd.date_range("2024-01-01", periods=100, freq="H")
    df = pd.DataFrame(
        {
            "item_id": "series_1",
            "timestamp": dates,
            "target": np.sin(np.arange(100) / 10.0),
            "temp": 20 + 5 * np.cos(np.arange(100) / 24.0),  # dynamic numerical covariate
            "holiday": (np.arange(100) % 24 == 0).astype(int),  # dynamic categorical covariate
        }
    )

    # Past values used for context
    context_df = df.iloc[:-24]
    # Future covariate values for the horizon (target values are unknown in future)
    future_cov = {
        "item_id": "series_1",
        "timestamp": df.iloc[-24:]["timestamp"],
        "temp": df.iloc[-24:]["temp"],
        "holiday": df.iloc[-24:]["holiday"],
    }
    future_df = pd.DataFrame(future_cov)

    # Load pre‑trained Chronos‑2 pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2")
    # Forecast 24 steps ahead using past and future covariates
    prediction = pipe.predict_df(
        context_df=context_df,
        future_df=future_df,
        prediction_length=24,
        quantile_levels=[0.1, 0.5, 0.9],
        id_col="item_id",
        time_col="timestamp",
        target_col="target",
    )
    print(prediction.head())


def example_moirai_large_with_covariates():
    """Forecast using Salesforce Moirai‑1.1‑R‑large with dynamic covariates.

    The `uni2ts` library provides a wrapper around Moirai.  When
    creating the `PandasDataset`, you specify which columns are
    dynamic real covariates.  Then, when constructing the
    `MoiraiForecast`, pass `feat_dynamic_real_dim` and
    `past_feat_dynamic_real_dim` equal to the number of covariates
    observed in the dataset.  This ensures the model uses the
    covariates when forecasting【784419845319154†L345-L356】.  The small
    (2.0) version currently does not support covariates, so the
    dimensions would be zero.
    """
    try:
        from uni2ts.data import PandasDataset
        from uni2ts.models import MoiraiForecast
    except ImportError:
        raise RuntimeError("uni2ts library is not installed")

    # Create a synthetic DataFrame for demonstration
    dates = pd.date_range("2024-01-01", periods=200, freq="H")
    df = pd.DataFrame(
        {
            "item_id": "series_1",
            "date": dates,
            "target": np.sin(np.arange(200) / 10.0),
            "temp": 20 + 5 * np.cos(np.arange(200) / 24.0),
            "is_holiday": (np.arange(200) % 24 == 0).astype(int),
        }
    )

    # Build dataset with dynamic real covariates
    dataset = PandasDataset.from_long_dataframe(
        df,
        item_id="item_id",
        time_col="date",
        target_col="target",
        feat_dynamic_real=["temp", "is_holiday"],
        past_feat_dynamic_real=None,
    )

    # Create Moirai forecast model
    model = MoiraiForecast.from_hyperparameters(
        prediction_length=24,
        context_length=168,
        patch_size=24,
        num_samples=64,
        target_dim=1,
        feat_dynamic_real_dim=dataset.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=dataset.num_past_feat_dynamic_real,
    )
    predictor = model.create_predictor()
    # Forecast on the last item in dataset
    test_ds = dataset.roll(roll_steps=1)[0]
    forecasts = predictor.predict(test_ds)
    print(forecasts.mean().head())


def example_tabpfn_ts_with_covariates():
    """Forecast using TabPFN‑TS with exogenous variables.

    TabPFN‑TS frames univariate forecasting as a tabular regression
    problem.  You first engineer features (time index, calendar
    features, lagged seasonal features) and concatenate your
    exogenous variables.  Then feed the resulting tabular data into
    the zero‑shot TabPFN regressor.  The README emphasises that
    exogenous variables are supported【289905141807667†L62-L63】.
    """
    try:
        from tabpfn_ts.tabpfn_ts import TabPFNZeroShotRegressor
        from tabpfn_ts.feature_transforms import (
            FeatureTransformer,
            RunningIndexFeature,
            CalendarFeature,
            AutoSeasonalFeature,
        )
    except ImportError:
        raise RuntimeError("tabpfn_ts library is not installed")

    # Synthetic univariate series and exogenous covariate (temperature)
    y = np.sin(np.arange(300) / 20.0)
    temp = 20 + 3 * np.random.randn(300)
    dates = pd.date_range("2023-01-01", periods=300, freq="H")

    # Build feature transformer including auto‑seasonal and calendar features
    ft = FeatureTransformer(
        features=[RunningIndexFeature(), CalendarFeature(), AutoSeasonalFeature()]
    )
    # Prepare training windows
    context_length = 200
    horizon = 24
    X_train, y_train = [], []
    for i in range(50):
        start = i
        end = start + context_length
        X_slice = y[start:end]
        temp_slice = temp[start:end]
        X_train.append(
            {
                "series": X_slice,
                "exogenous": temp_slice,
                "timestamp": dates[start:end],
            }
        )
        y_train.append(y[end : end + horizon])

    # Transform features to tabular format; include exogenous variable
    X_tab = []
    for window in X_train:
        # Transform time features
        feats = ft.transform(window["series"], freq="H")
        # Concatenate exogenous feature as an additional column
        exog = window["exogenous"].reshape(-1, 1)
        X_tab.append(np.hstack([feats, exog]))
    X_tab = np.array(X_tab)
    y_train = np.array(y_train)

    # Fit zero‑shot regressor (no training required; uses pre‑trained TabPFN model)
    model = TabPFNZeroShotRegressor.from_pretrained("PriorLabs/TabPFN-TS")
    # The model expects input of shape (n_samples, n_timesteps, n_features)
    preds = model.predict(X_tab, horizon)
    print(preds[:5])


def example_timesfm_with_covariates():
    """Forecast using Google TimesFM with dynamic covariates via `forecast_with_covariates`.

    As shown in the Databricks blog, TimesFM can ingest dynamic
    numerical and categorical covariates together with optional static
    covariates using the `forecast_with_covariates` function.  In the
    example, the pollutant concentration is predicted using temperature
    as a dynamic numerical covariate and wind direction as a static
    categorical covariate【123705836680735†screenshot】.
    """
    try:
        import timesfm
    except ImportError:
        raise RuntimeError("timesfm library is not installed")

    # Load model hyper‑parameters and checkpoint
    hparams = timesfm.TimesFmHparams(
        context_len=512,
        horizon_len=24,
        input_patch_len=32,
        output_patch_len=32,
        num_layers=4,
        model_dims=512,
        backend="cpu",
    )
    checkpoint = timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.5-200m-pytorch"
    )
    model = timesfm.TimesFm(hparams, checkpoint=checkpoint)

    # Example input series: predict pollution concentration
    # Each entry in `inputs` is a list of historical target values
    inputs = [list(np.random.randn(200))]
    # Dynamic numerical covariates dictionary maps name to a list of values per series
    dynamic_numerical_covariates = {"temp": [list(15 + 5 * np.random.randn(224))]}
    # Static categorical covariates map name to list of categories per series
    static_categorical_covariates = {"wnd_dir": [["NW"]]}
    # Forecast; note that freq can be specified as required; we pass dummy zeros
    cov_forecast, baseline_forecast = model.forecast_with_covariates(
        inputs=inputs,
        dynamic_numerical_covariates=dynamic_numerical_covariates,
        dynamic_categorical_covariates={},
        static_numerical_covariates={},
        static_categorical_covariates=static_categorical_covariates,
        freq=[0] * len(inputs[0]),
        xreg_mode="xreg + timesfm",
        force_on_cpu=False,
        normalize_xreg_target_per_input=True,
    )
    print(cov_forecast[0][:10])


def example_toto_with_covariates():
    """Forecast using Datadog Toto (TOT) treating covariates as additional channels.

    Toto’s `MaskedTimeseries` class expects a multi‑channel tensor of
    shape `(channels, time_steps)` and returns predictions for each
    channel.  Covariates can be included as additional channels in
    this tensor.  The Hugging Face model card demonstrates how to
    prepare the input and call the forecaster【318078313883841†L190-L235】.
    """
    try:
        import torch
        from tot import TotForecaster
    except ImportError:
        raise RuntimeError("tot package is not installed")

    # Suppose we have one target channel and one covariate channel
    target = np.sin(np.linspace(0, 2 * np.pi, 200))
    cov = 20 + 2 * np.cos(np.linspace(0, 4 * np.pi, 200))
    # Stack channels into tensor (channels, time_steps)
    inputs = torch.tensor(np.stack([target, cov], axis=0)).float().unsqueeze(0)  # (batch, channels, time_steps)
    # Load pre‑trained Toto forecaster
    forecaster = TotForecaster.from_pretrained("Datadog/Toto-Open-Base-1.0")
    preds = forecaster.forecast(inputs, prediction_length=24)
    print(preds.shape)


def example_flowstate_with_covariates():
    """Forecast using IBM FlowState with multivariate input and covariates.

    Flowstate accepts a 3‑D tensor of shape `(context_length, batch_size, n_channels)`
    and forecasts the future horizon.  You can include dynamic covariates as
    separate channels.  The model card example shows loading the model
    and calling it with a scale factor and prediction length【909980817167568†L97-L139】.
    """
    try:
        import torch
        from tsfm_public.models.flowstate import FlowStateForPrediction
    except ImportError:
        raise RuntimeError("tsfm_public library is not installed")

    # Create synthetic multivariate input with target and a covariate
    context_length = 128
    target = np.sin(np.linspace(0, 4 * np.pi, context_length))
    cov = np.cos(np.linspace(0, 4 * np.pi, context_length))
    # Shape should be (context_length, batch_size, channels)
    time_series = torch.tensor(
        np.stack([target, cov], axis=1)  # shape (context_length, channels)
    ).float().unsqueeze(1)
    # Load pre‑trained FlowState model
    model = FlowStateForPrediction.from_pretrained("ibm-research/flowstate")
    # Choose a scale factor based on the units of your data (here 1.0)
    pred = model(time_series, scale_factor=1.0, prediction_length=24)
    print(pred.shape)


def example_tinytimemixer_with_covariates():
    """Forecast using IBM Tiny Time Mixer with exogenous variables.

    The `TinyTimeMixerForecaster` from `sktime` fits on a target series
    with optional exogenous variables provided via `X`.  During
    prediction, you supply future exogenous variables.  The
    documentation provides a full example【195605334140760†L1093-L1123】.
    """
    try:
        from sktime.forecasting.tinytimemixer import TinyTimeMixerForecaster
        from sktime.datasets import load_longley
        from sktime.forecasting.model_selection import temporal_train_test_split
        from sktime.utils import fh as fh_utils
    except ImportError:
        raise RuntimeError("sktime library is not installed")

    # Use the Longley dataset (economic indicators) as an example
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    # Forecast horizon (next len of test set)
    fh = fh_utils.FH(np.arange(1, len(y_test) + 1))
    # Initialize the forecaster; `exog_dim` is the number of exogenous variables
    forecaster = TinyTimeMixerForecaster(
        input_chunk_length=24,
        output_chunk_length=len(fh),
        n_layers=2,
        target_dim=1,
        exog_dim=X_train.shape[1],
    )
    forecaster.fit(y_train, X=X_train)
    y_pred = forecaster.predict(fh, X=X_test)
    print(y_pred.head())


def example_patchtst_with_covariates():
    """Forecast using IBM PatchTST by concatenating covariates as channels.

    The PatchTST foundation model is channel‑independent: each channel
    corresponds to a univariate series and all channels share the same
    transformer weights【514347518787489†L84-L97】.  Although the base
    implementation does not expose a dedicated exogenous interface, you
    can include dynamic covariates as additional channels alongside
    the target.  Below is a simple usage example using the Hugging
    Face `PatchTSTForPrediction` class.  You need to adjust
    `context_length` and `prediction_length` to match the pre‑training.
    """
    try:
        import torch
        from transformers import PatchTSTForPrediction
    except ImportError:
        raise RuntimeError("transformers library with time‑series support is not installed")

    # Synthetic target and covariate
    context_length = 512
    target = np.sin(np.linspace(0, 10 * np.pi, context_length))
    cov1 = np.cos(np.linspace(0, 10 * np.pi, context_length))
    cov2 = np.random.randn(context_length) * 0.1
    # Stack into (batch, channels, sequence_length)
    x = torch.tensor(np.stack([target, cov1, cov2], axis=0)).float().unsqueeze(0)
    model = PatchTSTForPrediction.from_pretrained("ibm-granite/granite-timeseries-patchtst")
    # Generate prediction for the next 96 steps
    outputs = model(x, prediction_length=96)
    # The model returns predictions per channel; use the first channel as the forecast
    forecast = outputs.predictions[0, 0].detach().numpy()
    print(forecast[:10])


def example_patchtsmixer_with_covariates():
    """Forecast using IBM PatchTSMixer with covariates as additional channels.

    PatchTSMixer is a lightweight MLP‑Mixer based model for time‑series.
    As with PatchTST, dynamic covariates can be included as extra
    channels.  This example assumes an API similar to other
    Hugging Face time‑series models and illustrates the general data
    preparation process.  Replace `PatchTSMixerForPrediction` with
    the actual class if available.
    """
    try:
        import torch
        # PatchTSMixerForPrediction may reside in tsfm_public.models or transformers
        from tsfm_public.models.patchtsmixer import PatchTSMixerForPrediction
    except ImportError:
        raise RuntimeError(
            "PatchTSMixer implementation not found.  Ensure the tsfm_public library is installed."
        )

    context_length = 512
    target = np.sin(np.linspace(0, 20 * np.pi, context_length))
    covariate = np.random.randn(context_length)
    x = torch.tensor(np.stack([target, covariate], axis=0)).float().unsqueeze(0)
    model = PatchTSMixerForPrediction.from_pretrained("ibm/patchtsmixer-base-prediction-etth1")
    preds = model(x, prediction_length=96)
    forecast = preds.predictions[0, 0].detach().numpy()
    print(forecast[:10])


if __name__ == "__main__":
    # Uncomment the function you want to run.  Note: running these
    # examples requires that the corresponding libraries are installed and
    # may require significant memory.  They are provided for
    # illustration purposes only.

    # example_chronos2_with_covariates()
    # example_moirai_large_with_covariates()
    # example_tabpfn_ts_with_covariates()
    # example_timesfm_with_covariates()
    # example_toto_with_covariates()
    # example_flowstate_with_covariates()
    # example_tinytimemixer_with_covariates()
    # example_patchtst_with_covariates()
    # example_patchtsmixer_with_covariates()
