import pandas as pd
import torch
import os

# --- 1. Configurações ---
DATA_DIR = "../../data"
HORIZONTE_PREVISAO = 14

# --- 2. Carregar Dados ---
print("Carregando dados...")
hist_path = os.path.join(DATA_DIR, "hist.parquet")
future_cov_path = os.path.join(DATA_DIR, "future_covariates.parquet")

df_hist = pd.read_parquet(hist_path)
df_future_cov = pd.read_parquet(future_cov_path)

# Renomear o índice para 'timestamp' para compatibilidade
df_hist.index.name = 'timestamp'
df_future_cov.index.name = 'timestamp'
df_hist.reset_index(inplace=True)
df_future_cov.reset_index(inplace=True)

print("Dados históricos carregados:")
print(df_hist.tail())
print("\nCovariáveis futuras carregadas:")
print(df_future_cov.head())

# --- 3. Definir dispositivo ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsando dispositivo: {device}")

# ==============================================================================
# TESTE 1: Amazon Chronos com Covariáveis
# ==============================================================================

try:
    # Chronos 2 é necessário para o suporte a covariáveis com dataframes
    from chronos import Chronos2Pipeline

    print("\n--- Testando Amazon Chronos com Covariáveis ---")

    # 1. Carregar o pipeline (modelo)
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        torch_dtype=torch.bfloat16,
    )

    # 2. Preparar dados para a API predict_df
    # Adicionar uma coluna de ID
    df_hist['id'] = 'series_1'
    df_future_cov['id'] = 'series_1'

    # Combinar dados históricos e covariáveis futuras num único dataframe
    # O target para as datas futuras será NaN, o que é esperado
    df_full = pd.concat([df_hist, df_future_cov], ignore_index=True)

    print(f"Formato do dataframe combinado: {df_full.shape}")

    # 3. Rodar a previsão
    print(f"Rodando previsão para {HORIZONTE_PREVISAO} passos...")
    pred_df = pipeline.predict_df(
        df=df_full,
        prediction_length=HORIZONTE_PREVISAO,
        id_column="id",
        timestamp_column="timestamp",
        target="target",
        quantile_levels=[0.1, 0.5, 0.9]
    )

    print("Previsão concluída!")
    print(f"Formato do resultado da previsão: {pred_df.shape}")

    assert pred_df.shape[0] == HORIZONTE_PREVISAO, f'O tamanho da previsão ({pred_df.shape[0]}) não corresponde ao horizonte ({HORIZONTE_PREVISAO})'
    print("Tamanho da previsão verificado com sucesso.")
    print("Amostra da previsão:")
    print(pred_df.head())

    print("\nTeste do Chronos com covariáveis concluído.\n")

except ImportError:
    print("Chronos não instalado. Pulando teste.")
except Exception as e:
    print(f"Erro ao rodar Chronos: {e}")
