import pandas as pd
import torch
import os
import numpy as np

# --- 1. Configurações ---
DATA_DIR = "../../data"
HORIZONTE_PREVISAO = 14
CONTEXT_LENGTH = 96

# --- 2. Carregar Dados ---
print("Carregando dados...")
hist_path = os.path.join(DATA_DIR, "hist.parquet")
df_hist = pd.read_parquet(hist_path)
print("Dados históricos carregados.")

# --- 3. Definir dispositivo ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsando dispositivo: {device}")

# ==============================================================================
# TESTE 6: IBM Flowstate
# ==============================================================================

print("\n--- Testando IBM Flowstate ---")
print("\nAVISO: O modelo Flowstate está atualmente bloqueado.")
print("A API espera apenas um único canal de entrada ('single variate'),")
print("o que impede a inclusão de covariáveis através da abordagem de canal multivariado.")
print("Pulando a execução do Flowstate.")

# O código abaixo demonstra a tentativa falhada de usar múltiplos canais.
"""
try:
    from tsfm_public.models.flowstate import FlowStateForPrediction

    # 1. Carregar o modelo
    model = FlowStateForPrediction.from_pretrained("ibm-granite/granite-timeseries-flowstate-r1").to(device)

    # 2. Preparar dados
    # A tentativa de criar um tensor multicanal falha porque o modelo não o suporta.
    target_series = df_hist['target'].values[-CONTEXT_LENGTH:]
    day_of_week_series = df_hist['day_of_week'].values[-CONTEXT_LENGTH:]
    month_series = df_hist['month'].values[-CONTEXT_LENGTH:]

    input_series_np = np.stack([target_series, day_of_week_series, month_series], axis=1)
    input_tensor = torch.tensor(input_series_np, dtype=torch.float32).unsqueeze(1).to(device)

    print(f"Forma do tensor de entrada: {input_tensor.shape}")

    # 3. Rodar a previsão
    print(f"Rodando previsão para {HORIZONTE_PREVISAO} passos...")
    with torch.no_grad():
        forecast = model(
            input_tensor,
            prediction_length=HORIZONTE_PREVISAO,
            batch_first=False
        )

    # 4. Extrair a previsão
    median_prediction = forecast.prediction_outputs[0, 4, :, 0].cpu().numpy()

    print("Previsão concluída!")
    assert len(median_prediction) == HORIZONTE_PREVISAO
    print("Tamanho da previsão verificado com sucesso.")
    print("Amostra da previsão:")
    print(median_prediction[:5].round(2))

    print("\nTeste do Flowstate concluído.\n")

except ImportError:
    print("tsfm_public (granite-tsfm) não instalado. Pulando teste.")
except Exception as e:
    print(f"Erro ao rodar Flowstate: {e}")
"""
