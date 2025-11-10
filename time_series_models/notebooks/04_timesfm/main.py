import pandas as pd
import torch
import numpy as np
import os

# --- 1. Configurações ---
DATA_DIR = "../../data"
HORIZONTE_PREVISAO = 14
CONTEXT_LENGTH = 96 # O contexto deve ser um múltiplo de 32

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
# TESTE 4: Google TimesFM com Covariáveis
# ==============================================================================

try:
    import timesfm

    print("\n--- Testando Google TimesFM com Covariáveis ---")

    if len(df_hist) < CONTEXT_LENGTH:
        raise ValueError(f"TimesFM precisa de pelo menos {CONTEXT_LENGTH} pontos de dados.")

    # 1. Carregar o modelo
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        device_map=device
    )

    # 2. Configurar a previsão
    forecast_config = timesfm.ForecastConfig(
        max_context=CONTEXT_LENGTH,
        max_horizon=HORIZONTE_PREVISAO,
        normalize_inputs=True,
        use_continuous_quantile_head=False,
        return_backcast=True, # Necessário para XReg
    )
    model.compile(forecast_config)

    # 3. Preparar dados de entrada
    context_data = df_hist['target'].values[-CONTEXT_LENGTH:]

    df_full_cov = pd.concat([df_hist[['day_of_week', 'month']], df_future_cov[['day_of_week', 'month']]])
    covariates_data = df_full_cov[-(CONTEXT_LENGTH + HORIZONTE_PREVISAO):].to_numpy()

    dynamic_numerical_covariates = {
        'day_of_week': [covariates_data[:, 0]],
        'month': [covariates_data[:, 1]]
    }

    # 4. Rodar a previsão com covariáveis
    print(f"Rodando previsão para {HORIZONTE_PREVISAO} passos...")
    point_forecast, _ = model.forecast_with_covariates(
        inputs=[context_data],
        dynamic_numerical_covariates=dynamic_numerical_covariates
    )

    forecast_values = point_forecast[0]

    print("Previsão concluída!")
    assert len(forecast_values) == HORIZONTE_PREVISAO, f"O tamanho da previsão ({len(forecast_values)}) não corresponde ao horizonte ({HORIZONTE_PREVISAO})"
    print("Tamanho da previsão verificado com sucesso.")
    print("Amostra da previsão:")
    print(forecast_values[:5].round(2))

    print("\nTeste do TimesFM concluído.\n")

except ImportError:
    print("TimesFM não instalado. Pulando teste.")
except Exception as e:
    print(f"Erro ao rodar TimesFM: {e}")
