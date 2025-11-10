import pandas as pd
import torch
import os
from gluonts.dataset.pandas import PandasDataset

# --- 1. Configurações ---
DATA_DIR = "../../data"
HORIZONTE_PREVISAO = 14
CONTEXT_LENGTH = 100 # Moirai precisa de um comprimento de contexto definido
FREQ = "D"

# --- 2. Carregar Dados ---
print("Carregando dados...")
hist_path = os.path.join(DATA_DIR, "hist.parquet")
future_cov_path = os.path.join(DATA_DIR, "future_covariates.parquet")

df_hist = pd.read_parquet(hist_path)
df_future_cov = pd.read_parquet(future_cov_path)

# --- 3. Preparar Dados para GluonTS ---
df_hist.index.name = 'timestamp'
df_future_cov.index.name = 'timestamp'
df_hist.reset_index(inplace=True)
df_future_cov.reset_index(inplace=True)

df_hist['item_id'] = 'series_1'
df_future_cov['item_id'] = 'series_1'

df_full = pd.concat([df_hist, df_future_cov], ignore_index=True)
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

dataset = PandasDataset.from_long_dataframe(
    df_full,
    target='target',
    item_id='item_id',
    timestamp='timestamp',
    freq=FREQ,
    feat_dynamic_real=['day_of_week', 'month']
)

print("Dados históricos e covariáveis futuras carregados e preparados.")
print(f"Número de covariáveis: {dataset.num_feat_dynamic_real}")

# --- 4. Definir dispositivo ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsando dispositivo: {device}")

# ==============================================================================
# TESTE 2: Salesforce Moirai
# ==============================================================================

try:
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    print("\n--- Testando Salesforce Moirai com Covariáveis ---")

    module = MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small")

    model = MoiraiForecast(
        module=module,
        prediction_length=HORIZONTE_PREVISAO,
        context_length=CONTEXT_LENGTH,
        target_dim=1,
        feat_dynamic_real_dim=dataset.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=0
    )

    predictor = model.create_predictor(batch_size=32)

    print(f"Rodando previsão para {HORIZONTE_PREVISAO} passos...")
    forecasts = predictor.predict(dataset)

    forecast_item = next(iter(forecasts))
    mean_forecast = forecast_item.mean

    print("Previsão concluída!")
    assert len(mean_forecast) == HORIZONTE_PREVISAO, f"O tamanho da previsão ({len(mean_forecast)}) não corresponde ao horizonte ({HORIZONTE_PREVISAO})"
    print("Tamanho da previsão verificado com sucesso.")
    print("Amostra da previsão (média):")
    print(mean_forecast[:5])

    print("\nTeste do Moirai concluído.\n")

except ImportError:
    print("Moirai (uni2ts) não instalado. Pulando teste.")
except Exception as e:
    print(f"Erro ao rodar Moirai: {e}")
