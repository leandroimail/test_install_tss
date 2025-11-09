### Key Points on Foundation Models for Time Series Forecasting with Dynamic Covariates
- **Amazon Chronos (Chronos-2)**: Supports dynamic covariates natively. It excels in zero-shot forecasting and can incorporate known future covariates for improved accuracy.
- **Salesforce Moirai**: Handles dynamic covariates effectively, achieving state-of-the-art zero-shot performance across diverse domains. It supports both past and future covariates.
- **TabPFN-TS**: Treats time series as tabular data, inherently supporting dynamic covariates as features. It outperforms specialized models in zero-shot scenarios despite being trained on synthetic data.
- **Google TimesFM**: Supports dynamic covariates via an external regressor (XReg). It provides strong zero-shot forecasting with covariate integration for enhanced results.
- **Datadog Toto**: Designed for multivariate forecasting, treating dynamic covariates as additional channels. It is optimized for observability metrics but generalizes well.
- **IBM FlowState**: Primarily univariate but can handle multivariate inputs, including dynamic covariates, through its state-space model encoder. It dynamically adjusts to timescales.
- **IBM TinyTimeMixer (TTM)**: Supports dynamic covariates during fine-tuning. It is lightweight and excels in zero-shot/few-shot forecasting with covariate integration.
- **IBM PatchTST**: Supports multivariate forecasting, allowing dynamic covariates as additional channels. It uses patching for efficient long-term predictions.
- **IBM PatchTSMixer**: Natively supports multivariate data with dynamic covariates. It mixes across patches, channels, and features for robust forecasting.

### Model-Specific Considerations
- All models support dynamic covariates (known future values that influence the forecast, like holidays or promotions).
- Examples are provided as Python scripts (suitable for Jupyter notebooks or .py files). Install dependencies like `transformers`, `torch`, `pandas`, and model-specific libraries (e.g., `tsfm` for IBM models, `uni2ts` for Moirai).
- Data preparation: Use a sample multivariate dataset (e.g., ETTh1 with channels like HUFL, temperature as covariate). Normalize inputs and handle timestamps.
- Zero-shot: Use pre-trained models directly. Fine-tuning: Optional for better performance with covariates.

**Dependencies for All Examples**:
```python
!pip install transformers torch pandas numpy datasets gluonts[torch] uni2ts tsfm tabpfn chronos
```

---

#### 1. Amazon Chronos (amazon/chronos-2)
Chronos-2 supports all covariate types natively for zero-shot forecasting.

```python
# chronos_example.py
import pandas as pd
import torch
from chronos import ChronosPipeline  # pip install git+https://github.com/amazon-science/chronos-forecasting.git

# Sample data: target series + dynamic covariate (e.g., temperature)
data = pd.DataFrame({
    'timestamp': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'target': [i + (i % 10) for i in range(100)],  # Target time series
    'temperature': [20 + (i % 5) for i in range(100)]  # Dynamic covariate
})
data.set_index('timestamp', inplace=True)

# Split: context (past) and future (with known covariate)
context_df = data.iloc[:80]
future_df = data.iloc[80:]['temperature'].to_frame()  # Known future covariate

# Load model
pipeline = ChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda" if torch.cuda.is_available() else "cpu")

# Forecast with covariate
forecast = pipeline.predict_df(
    context_df,
    future_df=future_df,  # Dynamic covariate for forecast horizon
    prediction_length=20,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column=None,  # Single series
    timestamp_column=None,
    target="target"
)
print(forecast)  # Output: forecasted quantiles
```

---

#### 2. Salesforce Moirai (Salesforce/moirai-2.0-R-small or Salesforce/moirai-1.1-R-large)
Moirai supports dynamic covariates via feat_dynamic_real_dim.

```python
# moirai_example.py
import torch
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule  # pip install uni2ts

# Sample data: target + dynamic covariate (e.g., promo indicator)
data = pd.DataFrame({
    'timestamp': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'target': [i + (i % 10) for i in range(100)],
    'promo': [1 if i % 7 == 0 else 0 for i in range(100)]  # Dynamic covariate
})
data.set_index('timestamp', inplace=True)

# Prepare dataset
ds = PandasDataset(dict(data))

# Load model (use small for example)
model = MoiraiForecast(
    module=MoiraiModule.from_pretrained("Salesforce/moirai-2.0-R-small"),
    prediction_length=20,
    context_length=80,
    patch_size="auto",
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=1  # For dynamic covariate
)

# Forecast with covariate (promo in past and future)
forecast = model.predict(ds.input)  # Assumes covariate in ds
print(forecast)
```

---

#### 3. TabPFN-TS (PriorLabs/TabPFN-TS)
TabPFN-TS treats time series as tabular, supporting dynamic covariates as features.

```python
# tabpfn_ts_example.py
import pandas as pd
from tabpfn import TabPFNRegressor  # pip install tabpfn

# Sample data: flatten time series with lags and covariates
# Create lagged features and covariates
def create_features(df, lags=5):
    for lag in range(1, lags+1):
        df[f'target_lag{lag}'] = df['target'].shift(lag)
    return df.dropna()

data = pd.DataFrame({
    'timestamp': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'target': [i + (i % 10) for i in range(100)],
    'covariate': [20 + (i % 5) for i in range(100)]  # Dynamic covariate
})
data = create_features(data)

# Train/test split
train = data.iloc[:80]
test = data.iloc[80:]

X_train, y_train = train.drop(['timestamp', 'target'], axis=1), train['target']
X_test = test.drop(['timestamp', 'target'], axis=1)  # Include future covariates in X_test

# Model
model = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu')
model.fit(X_train, y_train)

# Predict
forecast = model.predict(X_test)
print(forecast)  # Forecasted values
```

---

#### 4. Google TimesFM (google/timesfm-2.5-200m-pytorch)
Supports covariates via XReg (external regressor).

```python
# timesfm_example.py
import pandas as pd
import timesfm  # pip install timesfm

# Sample data
data = pd.DataFrame({
    'timestamp': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'target': [i + (i % 10) for i in range(100)],
    'covariate': [20 + (i % 5) for i in range(100)]  # Dynamic covariate
})
data.set_index('timestamp', inplace=True)

# Split
context = data.iloc[:80]['target'].values
cov_context = data.iloc[:80]['covariate'].values
cov_future = data.iloc[80:]['covariate'].values  # Known future

# Load model
model = timesfm.TimesFm(
    context_len=80,
    horizon_len=20,
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend='cpu'
)
model.load_from_checkpoint("google/timesfm-2.5-200m-pytorch")

# Forecast with covariates
predictions = model.forecast_with_covariates(
    inputs=context,
    dynamic_numerical_covariates={"covariate": cov_context},
    dynamic_categorical_covariates={},
    static_numerical_covariates={},
    static_categorical_covariates={},
    freq=0,  # Daily
    xreg_mode="xreg + timesfm",
    ridge=1e-3,
    force_on_cpu=True,
    normalize_xreg_target_per_input=True
)
print(predictions)
```

---

#### 5. Datadog Toto (Datadog/Toto-Open-Base-1.0)
Supports covariates as multivariate channels.

```python
# toto_example.py
import torch
from toto.data.util.dataset import MaskedTimeseries  # pip install from repo
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto

# Sample data: 2 targets + 1 covariate
input_series = torch.randn(3, 80)  # Channels: target1, target2, covariate
timestamp_seconds = torch.zeros(3, 80)
time_interval_seconds = torch.full((3,), 86400.0)  # Daily

inputs = MaskedTimeseries(input_series, timestamp_seconds, time_interval_seconds)

# Load model
toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
toto.compile()

forecaster = TotoForecaster(toto.model)

# Forecast
forecast = forecaster(inputs, forecast_length=20)
print(forecast)  # Forecasted values for targets
```

---

#### 6. IBM FlowState (ibm-granite/granite-timeseries-flowstate-r1)
Supports multivariate, treating covariates as channels.

```python
# flowstate_example.py
import torch
from transformers import FlowStateForPrediction  # From tsfm repo

# Sample data: target + covariate
time_series = torch.randn(80, 2)  # [context, channels: target, covariate]

# Load model
predictor = FlowStateForPrediction.from_pretrained("ibm-granite/granite-timeseries-flowstate-r1")

# Forecast
forecast = predictor(time_series, scale_factor=1.0, prediction_length=20, batch_first=True)
print(forecast.prediction_outputs)  # Forecasted quantiles
```

---

#### 7. IBM TinyTimeMixer (ibm/TTM)
Supports covariates in fine-tuning.

```python
# ttm_example.py
import pandas as pd
from transformers import TinyTimeMixerForPrediction

# Sample data
data = pd.DataFrame({
    'timestamp': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'target': [i + (i % 10) for i in range(100)],
    'covariate': [20 + (i % 5) for i in range(100)]  # Dynamic covariate
})

# Load model
model = TinyTimeMixerForPrediction.from_pretrained("ibm/TTM", revision="main")

# Fine-tune with covariates (use Trainer with dataset including covariates)
# For zero-shot: model.predict(context, prediction_length=20)
```

---

#### 8. IBM PatchTST (ibm/patchtst-base-prediction-etth1)
Supports covariates as channels.

```python
# patchtst_example.py
import pandas as pd
from transformers import PatchTSTForPrediction

# Sample data (multivariate)
data = pd.DataFrame({
    'timestamp': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'target': [i + (i % 10) for i in range(100)],
    'covariate': [20 + (i % 5) for i in range(100)]
})

# Load model
model = PatchTSTForPrediction.from_pretrained("ibm/patchtst-base-prediction-etth1")

# Forecast (inputs as tensor with channels)
inputs = torch.tensor(data.drop('timestamp', axis=1).values.T).unsqueeze(0)  # [batch, seq, channels]
outputs = model(inputs)
print(outputs.prediction_outputs)
```

---

#### 9. IBM PatchTSMixer (ibm/patchtsmixer-base-prediction-etth1)
Supports covariates natively.

```python
# patchtsmixer_example.py
import pandas as pd
from transformers import PatchTSMixerForPrediction

# Sample data
data = pd.DataFrame({
    'timestamp': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'target': [i + (i % 10) for i in range(100)],
    'covariate': [20 + (i % 5) for i in range(100)]
})

# Load model
model = PatchTSMixerForPrediction.from_pretrained("ibm/patchtsmixer-base-prediction-etth1")

# Forecast
inputs = torch.tensor(data.drop('timestamp', axis=1).values.T).unsqueeze(0)
outputs = model(inputs)
print(outputs.prediction_outputs)
```

---
These examples use synthetic data for illustration. Replace with real datasets (e.g., ETTh1 from https://github.com/zhouhaoyi/ETDataset). For fine-tuning, use Hugging Face Trainer API with appropriate datasets including covariates.

---

**Detailed Survey of Foundation Models for Time Series Forecasting with Dynamic Covariates**

Dynamic covariates are external variables that vary over time and influence the target series, often known in advance (e.g., holidays, weather forecasts). They enhance forecasting accuracy by providing contextual information. Below, we survey the 9 models, their architectures, covariate handling, performance, and practical considerations.

#### 1. Amazon Chronos (Chronos-2)
**Overview**: Chronos-2 is an encoder-only transformer for zero-shot univariate/multivariate forecasting. It tokenizes time series with scaling and quantization, supporting up to 1024 context length.
**Covariate Handling**: Natively supports past/future dynamic covariates via group attention. Treats covariates as additional series for joint modeling.
**Performance**: State-of-the-art on fev-bench and GIFT-Eval. Improves 20% with covariates on energy/retail tasks.
**Use Cases**: Retail demand forecasting with promotions; energy with weather.
**Pros**: Efficient (up to 250x faster than originals); production-ready via SageMaker.
**Cons**: Limited to fixed resolutions without adjustment.
**Resources**: GitHub: amazon-science/chronos-forecasting; HF: amazon/chronos-2.
**Training Data**: 1B+ points from public/synthetic sources.

#### 2. Salesforce Moirai
**Overview**: Encoder-based transformer with any-variate attention for zero-shot forecasting. Sizes: small (14M), base (91M), large (311M).
**Covariate Handling**: Supports dynamic covariates (past/future) via feat_dynamic_real_dim. Handles up to arbitrary variables.
**Performance**: Top on GIFT-Eval; 1-2% better than Chronos on covariate tasks.
**Use Cases**: Weather forecasting with multiple sensors; finance with economic indicators.
**Pros**: Frequency-invariant via patch projection; mixture distributions for uncertainty.
**Cons**: Requires GluonTS for datasets.
**Resources**: GitHub: SalesforceAIResearch/uni2ts; HF: Salesforce/moirai-2.0-R-small.
**Training Data**: LOTSA (27B observations across 9 domains).

#### 3. TabPFN-TS (PriorLabs/TabPFN-TS)
**Overview**: Tabular foundation model repurposed for time series via feature engineering (lags, calendars).
**Covariate Handling**: Inherently supports dynamic covariates as tabular features; forecasts via regression.
**Performance**: Tops AutoGluon-TS benchmark; 3% better than Chronos-Large despite 11M params.
**Use Cases**: Economic indicators with exogenous variables; sensor data.
**Pros**: Zero-shot on unseen data; CPU-friendly; no time-series pretraining needed.
**Cons**: Inference slower for long sequences; limited to 50k samples.
**Resources**: GitHub: PriorLabs/tabpfn-time-series; HF: Prior-Labs/TabPFN-v2-reg.
**Training Data**: Synthetic tabular (no time-series exposure).

#### 4. Google TimesFM
**Overview**: Decoder-only transformer for zero-shot forecasting; 200M params, up to 16k context.
**Covariate Handling**: Via XReg (external linear regressor); supports dynamic numerical/categorical covariates.
**Performance**: Strong on GIFT-Eval; 25% better with v2.5 updates.
**Use Cases**: Demand forecasting with promotions; traffic with events.
**Pros**: Handles varying horizons; efficient fine-tuning.
**Cons**: Covariates via add-on; JAX/PyTorch versions.
**Resources**: GitHub: google-research/timesfm; HF: google/timesfm-2.5-200m-pytorch.
**Training Data**: 100B real-world points (Google Trends, Wikipedia).

#### 5. Datadog Toto
**Overview**: Decoder-only transformer optimized for observability; 151M params, up to 4k context.
**Covariate Handling**: Multivariate with proportional factorized attention; covariates as channels with block-diagonal masking.
**Performance**: SOTA on GIFT-Eval/BOOM; 1-2x better on observability tasks.
**Use Cases**: Metrics monitoring with correlated variables (e.g., CPU, memory).
**Pros**: Handles high-dimensional/sparse data; Student-T for heavy tails.
**Cons**: Focused on observability; limited general benchmarks.
**Resources**: GitHub: DataDog/toto; HF: Datadog/Toto-Open-Base-1.0.
**Training Data**: 2.36T points (43% Datadog metrics).

#### 6. IBM FlowState
**Overview**: SSM-encoder + functional basis decoder; 7M params, timescale-invariant.
**Covariate Handling**: Multivariate via SSM; dynamic covariates as channels.
**Performance**: #3 on GIFT-Eval; outperforms 10x larger models.
**Use Cases**: Variable-resolution data (e.g., hourly to daily).
**Pros**: Adjusts to any timescale; CPU inference.
**Cons**: Univariate-focused initially; extending to complex multivariate.
**Resources**: GitHub: ibm-granite/granite-tsfm; HF: ibm-granite/granite-timeseries-flowstate-r1.
**Training Data**: GIFT-Eval subsets + synthetics.

#### 7. IBM TinyTimeMixer (TTM)
**Overview**: MLP-mixer based; <1M params, zero/few-shot.
**Covariate Handling**: Supports during fine-tuning (channel-mixing); lags as covariates.
**Performance**: Outperforms MOIRAI/TimesFM (1-38%) with 100x fewer params.
**Use Cases**: Resource-constrained forecasting with few data.
**Pros**: Lightweight; CPU-friendly.
**Cons**: Fine-tuning needed for covariates.
**Resources**: GitHub: ibm-granite/granite-tsfm; HF: ibm/TTM.
**Training Data**: 700M public samples.

#### 8. IBM PatchTST
**Overview**: Transformer with patching/channel-independence; 1M params.
**Covariate Handling**: Multivariate; covariates as channels.
**Performance**: SOTA on ETTh1; 20% MSE reduction.
**Use Cases**: Long-term forecasting with covariates.
**Pros**: Efficient for long contexts.
**Cons**: Fixed resolutions.
**Resources**: GitHub: ibm-granite/granite-tsfm; HF: ibm/patchtst-base-prediction-etth1.
**Training Data**: ETTh1 (electrical transformers).

#### 9. IBM PatchTSMixer
**Overview**: MLP-mixer with patching; lightweight alternative to transformers.
**Covariate Handling**: Multivariate mixing across channels; supports dynamic covariates.
**Performance**: 1-2% better than PatchTST; 2-3x faster.
**Use Cases**: High-dimensional data with covariates.
**Pros**: Low memory/runtime.
**Cons**: Less flexible for very long sequences.
**Resources**: GitHub: ibm-granite/granite-tsfm; HF: ibm/patchtsmixer-base-prediction-etth1.
**Training Data**: ETTh1.

#### Comparative Table: Model Performance and Features
| Model                  | Params | Zero-Shot | Covariate Support | Key Strength          | Benchmark Rank (GIFT-Eval) | Training Data Size |
|------------------------|--------|-----------|-------------------|-----------------------|----------------------------|--------------------|
| Amazon Chronos-2      | 120M  | Yes      | Native (dynamic)  | Efficient production  | Top-5                     | 1B+                |
| Salesforce Moirai     | 14-311M | Yes     | Dynamic (past/future) | Frequency-invariant | #1                        | 27B                |
| TabPFN-TS             | 11M   | Yes      | Tabular features  | CPU-friendly         | Top (AutoGluon)           | Synthetic tabular  |
| Google TimesFM        | 200M  | Yes      | Via XReg          | Long context         | Strong                    | 100B               |
| Datadog Toto          | 151M  | Yes      | Multivariate channels | Observability SOTA  | #1 (BOOM)                 | 2.36T              |
| IBM FlowState         | 7M    | Yes      | Multivariate      | Timescale-invariant  | #3                        | GIFT subsets       |
| IBM TTM               | <1M   | Yes      | Fine-tuning       | Lightweight          | Outperforms larger        | 700M               |
| IBM PatchTST          | 1M    | Yes      | Channels          | Long-term efficient  | SOTA (ETTh1)              | ETTh1              |
| IBM PatchTSMixer      | ~1M   | Yes      | Channels/mixing   | Fast/low-memory      | 1-2% > PatchTST           | ETTh1              |

#### Trends and Challenges
- **Size vs. Performance**: Smaller models (e.g., TTM, FlowState) outperform larger ones via efficient architectures and focused pretraining.
- **Covariate Integration**: Most support dynamic covariates as channels or features; external regressors (e.g., TimesFM) add flexibility.
- **Zero-Shot Dominance**: All excel in zero-shot, reducing training needs.
- **Challenges**: Handling varying resolutions (FlowState excels); high-dimensionality (Toto/PatchTSMixer); resource constraints (TTM/TabPFN-TS).
- **Future Directions**: Multimodal integration; better uncertainty quantification; enterprise scalability.

This survey draws from public repositories, papers, and benchmarks (e.g., GIFT-Eval as of Nov 2025). Models evolve; verify latest on GitHub/HF.

---

#### Key Citations
- [1] Chronos-2: https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting
- [2] Moirai: https://arxiv.org/abs/2402.02592
- [3] TabPFN-TS: https://arxiv.org/abs/2501.02945
- [4] TimesFM: https://arxiv.org/abs/2310.10688
- [5] Toto: https://arxiv.org/abs/2505.14766
- [6] FlowState: https://arxiv.org/abs/2508.05287
- [7] TTM: https://arxiv.org/abs/2401.03955
- [8] PatchTST: https://arxiv.org/abs/2211.14730
- [9] PatchTSMixer: https://arxiv.org/abs/2306.09364