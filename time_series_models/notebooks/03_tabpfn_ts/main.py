import pandas as pd
import torch
import os
import warnings

# --- 1. Configurações ---
DATA_DIR = "../../data"
HORIZONTE_PREVISAO = 14

warnings.filterwarnings('ignore', category=UserWarning)

# --- 2. Carregar Dados ---
print("Carregando dados...")
hist_path = os.path.join(DATA_DIR, "hist.parquet")
future_cov_path = os.path.join(DATA_DIR, "future_covariates.parquet")

df_hist = pd.read_parquet(hist_path)
df_future_cov = pd.read_parquet(future_cov_path)

# --- 3. Preparar Dados ---
df_hist.index.name = 'timestamp'
df_future_cov.index.name = 'timestamp'
df_hist.reset_index(inplace=True)
df_future_cov.reset_index(inplace=True)

df_hist['item_id'] = 'series_1'
df_future_cov['item_id'] = 'series_1'

print("Dados históricos e de covariáveis futuras carregados.")

# ==============================================================================
# TESTE 3: TabPFN-TS
# ==============================================================================

print("\n--- Testando TabPFN-TS com Covariáveis ---")
print("\nAVISO: O modelo TabPFN-TS está atualmente bloqueado neste ambiente.")
print("O modo 'CLIENT' excede o tempo limite de rede/processamento.")
print("O modo 'LOCAL' requer uma GPU, que não está disponível.")
print("Pulando a execução do TabPFN-TS.")

# O código abaixo é a implementação correta, mas não pode ser executado.
"""
try:
    from tabpfn_time_series import TabPFNTimeSeriesPredictor, TabPFNMode, TimeSeriesDataFrame

    # 1. Converter para TimeSeriesDataFrame
    train_tsdf = TimeSeriesDataFrame.from_data_frame(
        df_hist,
        timestamp_column="timestamp",
        id_column="item_id",
    )
    test_tsdf = TimeSeriesDataFrame.from_data_frame(
        df_future_cov,
        timestamp_column="timestamp",
        id_column="item_id",
    )

    # 2. Inicializar o preditor
    predictor = TabPFNTimeSeriesPredictor(
        tabpfn_mode=TabPFNMode.CLIENT,
        config={'prediction_length': HORIZONTE_PREVISAO, 'N_ensemble_configurations': 4}
    )

    # 3. Prever
    print(f"Rodando 'predict' para {HORIZONTE_PREVISAO} passos...")
    pred_tsdf = predictor.predict(train_tsdf, test_tsdf)
    pred_df = pred_tsdf.to_pandas()


    print("Previsão concluída!")
    assert pred_df.shape[0] == HORIZONTE_PREVISAO, f"O tamanho da previsão ({pred_df.shape[0]}) não corresponde ao horizonte ({HORIZONTE_PREVISAO})"
    print("Tamanho da previsão verificado com sucesso.")
    print("Amostra da previsão:")
    print(pred_df.head())

    print("\nTeste do TabPFN-TS concluído.\n")

except ImportError as e:
    print(f"Erro de importação detalhado: {e}")
except Exception as e:
    print(f"Erro ao rodar TabPFN-TS: {e}")
"""
