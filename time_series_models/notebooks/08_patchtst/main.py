# =====================================================================================
# ATENÇÃO: BLOQUEIO DE AMBIENTE
# =====================================================================================
# A lógica para manipulação de dados e uso de covaráveis com o PatchTST
# via AutoGluon está implementada abaixo. No entanto, o script pode falhar
# no ambiente atual devido a um problema de acesso à rede para download de
# componentes ou modelos necessários para o AutoGluon.
# =====================================================================================

import pandas as pd
import matplotlib.pyplot as plt
import traceback
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

def main():
    """
    Função principal para treinar o modelo PatchTST com covaráveis usando AutoGluon,
    fazer previsões e plotar os resultados.
    """
    try:
        # Carregar dados
        print("Carregando dados...")
        df_hist = pd.read_parquet('../../data/hist.parquet')
        df_future_cov = pd.read_parquet('../../data/future_covariates.parquet')

        # Preparar dados
        # AutoGluon requer uma coluna 'item_id'
        df_hist['item_id'] = 'T1'
        df_future_cov['item_id'] = 'T1'

        # Criar índice de tempo para os dados históricos
        end_time_hist = pd.to_datetime('2024-07-15 23:00:00')
        start_time_hist = end_time_hist - pd.DateOffset(hours=len(df_hist) - 1)
        df_hist['timestamp'] = pd.date_range(start=start_time_hist, periods=len(df_hist), freq='h')

        # Combinar dados históricos e futuros para criar known_covariates
        # Renomear 'target' para 'value' para consistência, embora o nome não seja crítico aqui
        df_hist.rename(columns={'target': 'value'}, inplace=True)

        # Preparar dataframe futuro com as covaráveis conhecidas
        start_time_future = end_time_hist + pd.DateOffset(hours=1)
        df_future_cov['timestamp'] = pd.date_range(start=start_time_future, periods=len(df_future_cov), freq='h')

        # Juntar os dataframes para criar o conjunto de dados completo para AutoGluon
        # As covaráveis futuras precisam estar alinhadas com o horizonte de previsão
        known_covariates = pd.concat([df_hist[['item_id', 'timestamp', 'day_of_week', 'month']],
                                      df_future_cov[['item_id', 'timestamp', 'day_of_week', 'month']]])

        # O `TimeSeriesDataFrame` para treino contém apenas os dados históricos
        train_data = TimeSeriesDataFrame.from_data_frame(
            df_hist,
            id_column="item_id",
            timestamp_column="timestamp"
        )

        # Parâmetros
        prediction_length = len(df_future_cov)

        # Instanciar e treinar o modelo
        print("\nInstanciando e treinando o modelo...")
        predictor = TimeSeriesPredictor(
            prediction_length=prediction_length,
            target="value",
            known_covariates_names=["day_of_week", "month"],
            eval_metric="MASE",
        )

        # Especificar o modelo PatchTST
        hyperparameters = {
            "PatchTST": {}
        }

        predictor.fit(
            train_data,
            hyperparameters=hyperparameters,
            time_limit=300 # Tempo limite para o treino
        )

        # Fazer a previsão
        print("Fazendo a previsão...")
        predictions = predictor.predict(train_data, known_covariates=known_covariates)

        # Visualizar resultados
        print("Plotando resultados...")
        plt.figure(figsize=(12, 6))

        y_past = train_data.loc["T1"]["value"]
        y_pred = predictions.loc["T1"]

        plt.plot(y_past.index, y_past, label='Histórico (Alvo)')
        plt.plot(y_pred.index, y_pred['mean'], label='Previsão', color='red')
        plt.title('Previsão PatchTST com Covariáveis (via AutoGluon)')
        plt.xlabel('Data')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)

        # Salvar o gráfico
        plt.savefig('patchtst_autogluon_forecast.png')
        print("Gráfico salvo como patchtst_autogluon_forecast.png")

    except Exception as e:
        print(f"Ocorreu um erro detalhado:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
