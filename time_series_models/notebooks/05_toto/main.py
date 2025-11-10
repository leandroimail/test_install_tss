import pandas as pd
import torch
import os
import numpy as np

# --- 1. Configurações ---
DATA_DIR = "../../data"
HORIZONTE_PREVISAO = 14
N_DIAS_HISTORICO = 100

# --- 2. Carregar Dados ---
print("Carregando dados...")
hist_path = os.path.join(DATA_DIR, "hist.parquet")
future_cov_path = os.path.join(DATA_DIR, "future_covariates.parquet")

df_hist = pd.read_parquet(hist_path)
df_future_cov = pd.read_parquet(future_cov_path)

print("Dados históricos e de covariáveis futuras carregados.")

# --- 3. Definir dispositivo ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsando dispositivo: {device}")

# ==============================================================================
# TESTE 5: Datadog Toto com Covariáveis
# ==============================================================================

try:
    from toto.model.toto import Toto
    from toto.inference.forecaster import TotoForecaster
    from toto.data.util.dataset import MaskedTimeseries

    print("\n--- Testando Datadog Toto com Covariáveis ---")

    # 1. Carregar o modelo pré-treinado
    toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
    toto.to(device)
    forecaster = TotoForecaster(toto.model)

    # 2. Preparar dados como um tensor multicanal
    target_series = df_hist['target'].values
    day_of_week_series = df_hist['day_of_week'].values
    month_series = df_hist['month'].values

    input_series_np = np.vstack([target_series, day_of_week_series, month_series])
    input_series = torch.tensor(input_series_np, dtype=torch.float32).to(device)

    n_channels = input_series.shape[0]
    print(f"Número de canais de entrada: {n_channels}")

    inputs = MaskedTimeseries(
        series=input_series,
        padding_mask=torch.full_like(input_series, True, dtype=torch.bool),
        id_mask=torch.zeros_like(input_series),
        timestamp_seconds=torch.zeros_like(input_series),
        time_interval_seconds=torch.full((n_channels,), 60*60*24).to(device),
    )

    # 3. Rodar a previsão
    print(f"Rodando previsão para {HORIZONTE_PREVISAO} passos...")
    forecast = forecaster.forecast(
        inputs,
        prediction_length=HORIZONTE_PREVISAO,
        num_samples=256,
        samples_per_batch=256,
    )

    # 4. Extrair a previsão do primeiro item, primeiro canal
    # A forma é [n_items, n_canais, horizonte_previsao]
    median_prediction = forecast.median[0, 0, :].cpu().numpy()

    print("Previsão concluída!")
    assert len(median_prediction) == HORIZONTE_PREVISAO, f"O tamanho da previsão ({len(median_prediction)}) não corresponde ao horizonte ({HORIZONTE_PREVISAO})"
    print("Tamanho da previsão verificado com sucesso.")
    print("Amostra da previsão (mediana do canal de target):")
    print(median_prediction[:5].round(2))

    print("\nTeste do Toto concluído.\n")

except ImportError:
    print("Toto-ts não instalado. Pulando teste.")
except Exception as e:
    print(f"Erro ao rodar Toto: {e}")
