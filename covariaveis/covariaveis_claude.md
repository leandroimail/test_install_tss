# Guia Completo: Foundation Models para Time Series com Covariáveis Dinâmicas

Este guia apresenta exemplos práticos de como usar 9 foundation models para forecasting de séries temporais com **covariáveis dinâmicas**.

## Índice

1. [Amazon Chronos-2](#1-amazon-chronos-2) ✅
2. [Salesforce Moirai](#2-salesforce-moirai) ✅
3. [TabPFN-TS](#3-tabpfn-ts) ✅
4. [Google TimesFM 2.5](#4-google-timesfm-25) ✅
5. [Datadog Toto](#5-datadog-toto) ⚠️
6. [IBM TinyTimeMixer (TTM)](#6-ibm-tinytimemixer-ttm) ✅
7. [IBM PatchTST](#7-ibm-patchtst) ❌
8. [IBM PatchTSMixer](#8-ibm-patchtsmixer) ❌
9. [IBM Flowstate](#9-ibm-flowstate) ❌

**Legenda:**
- ✅ Suporte nativo para covariáveis dinâmicas
- ⚠️ Suporte limitado/indireto
- ❌ Sem suporte nativo na versão atual

---

## 1. Amazon Chronos-2

**Status:** ✅ Suporte completo para covariáveis past-only e known-future

### Instalação

```bash
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

### Exemplo com Covariáveis

```python
import pandas as pd
from chronos import Chronos2Pipeline

# Carregar o modelo
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2", 
    device_map="cuda"  # ou "cpu"
)

# Criar dados de exemplo
# Context: dados históricos com target e covariáveis
context_data = {
    'id': ['series_1'] * 100,
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='D'),
    'target': np.random.randn(100).cumsum(),
    'temperatura': np.random.uniform(15, 30, 100),
    'promocao': np.random.binomial(1, 0.2, 100)
}
context_df = pd.DataFrame(context_data)

# Future: valores futuros conhecidos das covariáveis
future_data = {
    'id': ['series_1'] * 24,
    'timestamp': pd.date_range('2024-04-10', periods=24, freq='D'),
    'temperatura': np.random.uniform(15, 30, 24),
    'promocao': np.random.binomial(1, 0.3, 24)
}
future_df = pd.DataFrame(future_data)

# Gerar previsões com covariáveis
pred_df = pipeline.predict_df(
    context_df,
    future_df=future_df,  # Covariáveis futuras conhecidas
    prediction_length=24,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="target"
)

print("Previsões:")
print(pred_df.head())
```

### Tipos de Covariáveis Suportadas

- **Past-only covariates**: Apenas valores históricos disponíveis
- **Known-future covariates**: Valores futuros conhecidos (ex: feriados, promoções)
- **Categorical covariates**: Variáveis categóricas

---

## 2. Salesforce Moirai

**Status:** ✅ Suporte para covariáveis através de GluonTS

### Instalação

```bash
pip install git+https://github.com/SalesforceAIResearch/uni2ts.git
```

### Exemplo com Covariáveis

```python
import torch
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

# Configurações
MODEL = "moirai-2.0"  # ou "moirai", "moirai-moe"
SIZE = "small"  # small, base, large
PDT = 20  # prediction length
CTX = 200  # context length
BSZ = 32  # batch size

# Criar dados de exemplo com covariáveis
data = {
    'unique_id': ['ts_1'] * 300,
    'ds': pd.date_range('2024-01-01', periods=300, freq='H'),
    'y': np.random.randn(300).cumsum(),
    'covariate_1': np.random.randn(300),
    'covariate_2': np.random.uniform(0, 1, 300)
}
df = pd.DataFrame(data)

# Criar dataset GluonTS com covariáveis dinâmicas
ds = PandasDataset.from_long_dataframe(
    df,
    target='y',
    item_id='unique_id',
    timestamp='ds',
    feat_dynamic_real=['covariate_1', 'covariate_2']  # Covariáveis
)

# Split train/test
train, test_template = split(ds, offset=-100)
test_data = test_template.generate_instances(
    prediction_length=PDT,
    windows=5
)

# Carregar modelo
module = MoiraiModule.from_pretrained(f"Salesforce/moirai-2.0-R-{SIZE}")
model = MoiraiForecast(
    module=module,
    prediction_length=PDT,
    context_length=CTX,
    patch_size="auto",
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=2  # Número de covariáveis
)

# Fazer previsões
predictor = model.create_predictor(batch_size=BSZ)
forecasts = predictor.predict(test_data)

# Visualizar resultados
for forecast in forecasts:
    print(f"Média: {forecast.mean}")
    print(f"Quantis: {forecast.quantile(0.1)}, {forecast.quantile(0.9)}")
```

---

## 3. TabPFN-TS

**Status:** ✅ Suporte através de feature engineering

### Instalação

```bash
pip install git+https://github.com/PriorLabs/tabpfn-time-series.git
```

### Exemplo com Covariáveis

```python
import pandas as pd
import numpy as np
from tabpfn_time_series import TimeSeriesDataFrame, TabPFNTimeSeriesPredictor
from tabpfn_time_series import TabPFNMode

# Criar dados com covariáveis exógenas
dates = pd.date_range('2024-01-01', periods=500, freq='D')
data = pd.DataFrame({
    'timestamp': dates,
    'item_id': 0,
    'target': np.random.randn(500).cumsum(),
    'day_of_week': dates.dayofweek,
    'is_weekend': dates.dayofweek.isin([5, 6]).astype(int),
    'temperature': np.random.uniform(15, 35, 500),
    'promotion': np.random.binomial(1, 0.15, 500)
})

# Criar TimeSeriesDataFrame
# TabPFN-TS automaticamente usa todas as colunas como features
tsdf = TimeSeriesDataFrame.from_data_frame(
    data,
    id_column='item_id',
    timestamp_column='timestamp'
)

# Split train/test
prediction_length = 30
train_tsdf = tsdf[:-prediction_length]
test_tsdf = tsdf[-prediction_length:]

# Criar e treinar preditor
predictor = TabPFNTimeSeriesPredictor(
    tabpfn_mode=TabPFNMode.CLIENT,  # ou TabPFNMode.LOCAL
    prediction_length=prediction_length,
)

# Fazer previsões (inclui automaticamente as covariáveis)
predictions = predictor.predict(train_tsdf, test_tsdf)

print("Previsões com covariáveis:")
print(predictions.head())
```

**Nota:** TabPFN-TS trata forecasting como um problema de regressão tabular, usando automaticamente todas as colunas como features, incluindo covariáveis exógenas.

---

## 4. Google TimesFM 2.5

**Status:** ✅ Suporte através de XReg (External Regressors)

### Instalação

```bash
pip install git+https://github.com/google-research/timesfm.git
# Para usar XReg:
pip install timesfm[xreg]
```

### Exemplo com Covariáveis (TimesFM 2.5)

```python
import numpy as np
import timesfm
import torch

# Configurar modelo
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

# Compilar modelo com configurações
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True
    )
)

# Dados de exemplo
context_length = 512
horizon = 24

# Série temporal principal (target)
target_series = np.random.randn(context_length).cumsum()

# Covariáveis dinâmicas (incluindo valores futuros)
# Forma: (context_length + horizon,)
covariate_1 = np.random.randn(context_length + horizon)
covariate_2 = np.random.uniform(0, 1, context_length + horizon)

# Preparar covariáveis para XReg
# XReg precisa de valores passados E futuros
dynamic_covariates = {
    'cov1': [covariate_1],
    'cov2': [covariate_2]
}

# Fazer previsão com covariáveis usando XReg
point_forecast, quantile_forecast = model.forecast_with_covariates(
    inputs=[target_series],
    dynamic_numerical_covariates=dynamic_covariates,
    horizon=horizon,
    freq=[0],  # frequência (0 para automático)
    xreg_mode="timesfm + xreg",  # ou "xreg + timesfm"
    ridge=0.0,
    normalize_xreg_target_per_input=True
)

print(f"Previsão pontual shape: {point_forecast.shape}")
print(f"Previsão quantílica shape: {quantile_forecast.shape}")
```

### XReg Modes

- **"timesfm + xreg"**: TimesFM prevê primeiro, depois ajusta com regressão externa
- **"xreg + timesfm"**: Regressão externa primeiro, TimesFM prevê o resíduo
- **"xreg"**: Apenas regressão externa

---

## 5. Datadog Toto

**Status:** ⚠️ Suporta multivariate mas sem covariáveis exógenas explícitas

### Instalação

```bash
git clone https://github.com/DataDog/toto.git
cd toto
pip install -r requirements.txt
```

### Exemplo Multivariate

```python
import torch
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Carregar modelo
toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0').to(DEVICE)
toto.compile()  # Compilar para melhor performance
forecaster = TotoForecaster(toto.model)

# Criar dados multivariados
# Toto pode prever múltiplas séries relacionadas simultaneamente
n_variables = 7
context_length = 4096

input_series = torch.randn(n_variables, context_length).to(DEVICE)
timestamp_seconds = torch.zeros(n_variables, context_length).to(DEVICE)
time_interval_seconds = torch.full((n_variables,), 60*15).to(DEVICE)

# Preparar inputs
inputs = MaskedTimeseries(
    series=input_series,
    padding_mask=torch.full_like(input_series, True, dtype=torch.bool),
    id_mask=torch.zeros_like(input_series),
    timestamp_seconds=timestamp_seconds,
    time_interval_seconds=time_interval_seconds,
)

# Gerar previsões
forecast = forecaster.forecast(
    inputs,
    prediction_length=336,
    num_samples=256,
    samples_per_batch=256,
)

# Acessar resultados
median_prediction = forecast.quantiles[0.5]  # Mediana
mean_prediction = forecast.mean

print(f"Previsão mediana shape: {median_prediction.shape}")
print(f"Previsão média shape: {mean_prediction.shape}")
```

**Limitação:** Toto atualmente não tem suporte explícito para covariáveis exógenas independentes, mas pode modelar múltiplas séries relacionadas simultaneamente (multivariate).

---

## 6. IBM TinyTimeMixer (TTM)

**Status:** ✅ Suporte através do Exogenous Mixer

### Instalação

```bash
pip install git+https://github.com/ibm-granite/granite-tsfm.git
```

### Exemplo com Covariáveis

```python
import pandas as pd
import numpy as np
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from transformers import Trainer, TrainingArguments

# Criar dados com covariáveis
n_samples = 1000
data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
    'id': ['ts_1'] * n_samples,
    'target': np.random.randn(n_samples).cumsum(),
    # Covariáveis conhecidas no futuro
    'hour_of_day': np.tile(np.arange(24), n_samples // 24 + 1)[:n_samples],
    'day_of_week': np.repeat(np.arange(7), 24)[:n_samples] % 7,
    # Covariáveis observadas (passadas e futuras)
    'planned_event': np.random.binomial(1, 0.1, n_samples),
})

# Configurar preprocessor
context_length = 512
prediction_length = 96

tsp = TimeSeriesPreprocessor(
    id_columns=['id'],
    timestamp_column='timestamp',
    target_columns=['target'],
    control_columns=['planned_event'],  # Covariáveis conhecidas no futuro
    observable_columns=['hour_of_day', 'day_of_week'],  # Covariáveis observáveis
    context_length=context_length,
    prediction_length=prediction_length,
    scaling=True
)

# Preparar dados
train_data, val_data, test_data = tsp.get_datasets(
    data,
    split_config={'train': 0.7, 'valid': 0.15, 'test': 0.15}
)

# Carregar modelo pré-treinado
model = TinyTimeMixerForPrediction.from_pretrained(
    "ibm-granite/granite-timeseries-ttm-r2",
    revision="main"
)

# Configurar para usar exogenous mixer
model.config.enable_forecast_channel_mixing = True  # Habilitar channel mixing
model.config.enable_exogenous = True  # Habilitar exogenous mixer

# Fine-tune com covariáveis
training_args = TrainingArguments(
    output_dir="./ttm_finetuned",
    per_device_train_batch_size=32,
    learning_rate=1e-4,
    num_train_epochs=10,
    logging_steps=100,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

trainer.train()

# Fazer previsões
predictions = trainer.predict(test_data)
print(f"Previsões shape: {predictions.predictions.shape}")
```

### Recursos do Exogenous Mixer

- **Control columns**: Covariáveis com valores futuros conhecidos
- **Observable columns**: Covariáveis que podem ser observadas mas não previstas
- **Channel mixing**: Captura correlações entre variáveis

---

## 7. IBM PatchTST

**Status:** ❌ Sem suporte nativo para covariáveis

### Instalação

```bash
pip install transformers
```

### Exemplo Básico (Sem Covariáveis)

```python
from transformers import PatchTSTConfig, PatchTSTForPrediction
import torch

# Configuração
config = PatchTSTConfig(
    num_input_channels=1,
    context_length=512,
    prediction_length=96,
    patch_length=16,
    stride=8,
)

# Modelo
model = PatchTSTForPrediction(config)

# Dados de entrada (batch_size, context_length, num_channels)
past_values = torch.randn(32, 512, 1)

# Previsão
outputs = model(past_values=past_values)
predictions = outputs.prediction_outputs

print(f"Previsões shape: {predictions.shape}")
```

**Nota:** Existe uma [feature request aberta](https://github.com/huggingface/transformers/issues/28611) para adicionar suporte a covariáveis. Alternativas:
- Usar PatchTST para a série principal e combinar com modelo externo para covariáveis
- Aguardar implementação futura

---

## 8. IBM PatchTSMixer

**Status:** ❌ Sem suporte nativo para covariáveis

### Instalação

```bash
pip install transformers
```

### Exemplo Básico (Sem Covariáveis)

```python
from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction
import torch

# Configuração
config = PatchTSMixerConfig(
    num_input_channels=3,  # Múltiplos canais (multivariate)
    context_length=512,
    prediction_length=96,
    patch_length=16,
    stride=8,
    mode="common_channel"  # ou "mix_channel"
)

# Modelo
model = PatchTSMixerForPrediction(config)

# Dados multivariados (batch_size, context_length, num_channels)
past_values = torch.randn(32, 512, 3)

# Previsão
outputs = model(past_values=past_values)
predictions = outputs.prediction_outputs

print(f"Previsões shape: {predictions.shape}")
```

**Nota:** Similar ao PatchTST, há uma [feature request](https://github.com/huggingface/transformers/issues/28611) para adicionar covariáveis. O modelo suporta múltiplos canais, mas não diferencia entre targets e covariáveis exógenas.

---

## 9. IBM Flowstate

**Status:** ❌ Zero-shot apenas, sem suporte para covariáveis

### Instalação

```bash
pip install git+https://github.com/ibm-granite/granite-tsfm.git
```

### Exemplo Zero-Shot

```python
import torch
from tsfm_public import FlowStateForPrediction

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Carregar modelo
predictor = FlowStateForPrediction.from_pretrained(
    "ibm-granite/granite-timeseries-flowstate-r1"
).to(DEVICE)

# Dados de entrada (context_length, batch_size, n_channels)
time_series = torch.randn((2048, 32, 1), device=DEVICE)

# Gerar previsão
# scale_factor ajusta a resolução temporal
forecast = predictor(
    time_series,
    scale_factor=0.25,  # Ajustar conforme a sazonalidade dos dados
    prediction_length=960,
    batch_first=False
)

# Resultados (batch, quantiles, forecast_length, n_channels)
predictions = forecast.prediction_outputs
print(f"Previsões shape: {predictions.shape}")
```

**Características:**
- **Time-scale adjustable**: Pode adaptar para diferentes resoluções temporais
- **Zero-shot apenas**: Não requer fine-tuning
- **Sem covariáveis**: Focado em previsão temporal pura

### Scale Factor Recomendado

Para dados com sazonalidade N=96 (ex: dados de 15min com ciclo diário):
```python
scale_factor = 32 / 96  # ≈ 0.33
```

---

## Comparação dos Modelos

| Modelo | Covariáveis | Multivariate | Fine-tuning | Zero-shot |
|--------|-------------|--------------|-------------|-----------|
| **Chronos-2** | ✅ Completo | ✅ | ✅ | ✅ |
| **Moirai** | ✅ Completo | ✅ | ✅ | ✅ |
| **TabPFN-TS** | ✅ Via features | ✅ | ❌ | ✅ |
| **TimesFM 2.5** | ✅ XReg | ✅ | ✅ | ✅ |
| **Toto** | ⚠️ Limitado | ✅ | ❌ | ✅ |
| **TTM** | ✅ Exog Mixer | ✅ | ✅ | ✅ |
| **PatchTST** | ❌ | ✅ | ✅ | ✅ |
| **PatchTSMixer** | ❌ | ✅ | ✅ | ✅ |
| **Flowstate** | ❌ | ✅ | ❌ | ✅ |

---

## Recomendações de Uso

### Escolha Chronos-2 se:
- Precisa de suporte completo para covariáveis past-only e known-future
- Quer interface simples via pandas DataFrame
- Deseja o melhor desempenho em benchmarks com covariáveis

### Escolha Moirai se:
- Já usa o ecossistema GluonTS
- Precisa de flexibilidade em frequências temporais
- Quer suporte robusto para any-variate

### Escolha TabPFN-TS se:
- Tem covariáveis categóricas complexas
- Prefere abordagem de regressão tabular
- Quer previsões rápidas sem GPU

### Escolha TimesFM 2.5 se:
- Precisa de controle fino sobre como usar covariáveis (XReg modes)
- Trabalha com contextos muito longos (até 16k)
- Quer previsões quantílicas contínuas

### Escolha TTM se:
- Precisa de fine-tuning eficiente (<1M parâmetros)
- Quer executar em CPU
- Valoriza velocidade de treinamento

### Escolha Toto se:
- Trabalha com dados de observabilidade/métricas
- Precisa lidar com dados esparsos e não-estacionários
- Foca em multivariate forecasting

---

## Exemplos Práticos Completos

### Exemplo 1: Previsão de Demanda com Promoções (Chronos-2)

```python
import pandas as pd
import numpy as np
from chronos import Chronos2Pipeline

# Simular dados de vendas com promoções
dates = pd.date_range('2023-01-01', periods=365, freq='D')
n = len(dates)

sales_data = pd.DataFrame({
    'id': ['product_A'] * n,
    'timestamp': dates,
    'sales': np.random.poisson(100, n) + dates.dayofweek.values * 10,
    'price': np.random.uniform(9.99, 19.99, n),
    'promotion': np.random.binomial(1, 0.15, n),
    'holiday': dates.day_name().isin(['Saturday', 'Sunday']).astype(int)
})

# Adicionar efeito de promoção nas vendas
sales_data['sales'] = sales_data['sales'] * (1 + sales_data['promotion'] * 0.5)

# Carregar modelo
pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2")

# Separar treino e covariáveis futuras
train_end = '2023-11-30'
train_data = sales_data[sales_data['timestamp'] <= train_end]
future_data = sales_data[sales_data['timestamp'] > train_end][
    ['id', 'timestamp', 'price', 'promotion', 'holiday']
]

# Previsão
forecasts = pipeline.predict_df(
    train_data,
    future_df=future_data,
    prediction_length=31,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column='id',
    timestamp_column='timestamp',
    target='sales'
)

print("Previsões de dezembro:")
print(forecasts)
```

### Exemplo 2: Consumo de Energia com Temperatura (Moirai)

```python
import pandas as pd
import numpy as np
from gluonts.dataset.pandas import PandasDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

# Simular dados de consumo de energia
hours = pd.date_range('2023-01-01', periods=8760, freq='H')  # 1 ano
temp = 20 + 10 * np.sin(2 * np.pi * np.arange(8760) / 8760)  # Temperatura
consumption = 50 + temp * 2 + np.random.randn(8760) * 5

energy_data = pd.DataFrame({
    'unique_id': ['grid_1'] * 8760,
    'ds': hours,
    'consumption': consumption,
    'temperature': temp,
    'hour': hours.hour,
    'is_weekend': hours.dayofweek.isin([5, 6]).astype(int)
})

# Criar dataset com covariáveis
ds = PandasDataset.from_long_dataframe(
    energy_data,
    target='consumption',
    item_id='unique_id',
    timestamp='ds',
    feat_dynamic_real=['temperature', 'hour', 'is_weekend']
)

# Modelo e previsão (similar ao exemplo anterior)
# ... código de modelagem ...
```

---

## Recursos Adicionais

### Documentação Oficial

- **Chronos-2**: [GitHub](https://github.com/amazon-science/chronos-forecasting)
- **Moirai**: [GitHub](https://github.com/SalesforceAIResearch/uni2ts)
- **TabPFN-TS**: [GitHub](https://github.com/PriorLabs/tabpfn-time-series)
- **TimesFM**: [GitHub](https://github.com/google-research/timesfm)
- **Toto**: [GitHub](https://github.com/DataDog/toto)
- **IBM TSFM**: [GitHub](https://github.com/ibm-granite/granite-tsfm)

### Papers

- Chronos-2: [arXiv:2510.15821](https://arxiv.org/abs/2510.15821)
- Moirai: [arXiv:2402.02592](https://arxiv.org/abs/2402.02592)
- TabPFN-TS: [NeurIPS 2024](https://arxiv.org/abs/2501.02945)
- TimesFM: [arXiv:2310.10688](https://arxiv.org/abs/2310.10688)
- Toto: [arXiv:2407.07874](https://arxiv.org/abs/2407.07874)
- TTM: [arXiv:2401.03955](https://arxiv.org/abs/2401.03955)

---

## Conclusão

Para trabalhar com **covariáveis dinâmicas**, recomendo especialmente:

1. **Chronos-2**: Melhor opção geral, interface simples, excelente desempenho
2. **Moirai**: Melhor para usuários GluonTS, muito flexível
3. **TTM**: Melhor para fine-tuning eficiente e execução em CPU

Os modelos PatchTST, PatchTSMixer e Flowstate são excelentes para outros casos de uso, mas atualmente não suportam covariáveis exógenas de forma nativa.