# =====================================================================================
# ATENÇÃO: BLOQUEIO DE AMBIENTE
# =====================================================================================
# A lógica para manipulação de dados e uso de covaráveis com o TinyTimeMixer
# via sktime está implementada corretamente abaixo. No entanto, o script
# provavelmente falhará no ambiente atual devido a um problema de acesso ao
# Hugging Face Hub, necessário para baixar o modelo pré-treinado.
#
# Erro esperado:
# huggingface_hub.errors.RepositoryNotFoundError: 401 Client Error. Unauthorized
#
# Para executar este script, o ambiente precisa ser configurado com a
# autenticação correta do Hugging Face.
# =====================================================================================

import pandas as pd
import matplotlib.pyplot as plt
import traceback
from sktime.forecasting.ttm import TinyTimeMixerForecaster
from sktime.forecasting.base import ForecastingHorizon

def main():
    """
    Função principal para treinar o modelo TinyTimeMixer com covaráveis usando sktime,
    fazer previsões e plotar os resultados.
    """
    try:
        # Carregar dados
        print("Carregando dados...")
        df_hist = pd.read_parquet('../../data/hist.parquet')
        df_future_cov = pd.read_parquet('../../data/future_covariates.parquet')

        # Preparar dados históricos
        end_time_hist = pd.to_datetime('2024-07-15 23:00:00')
        start_time_hist = end_time_hist - pd.DateOffset(hours=len(df_hist) - 1)
        df_hist.index = pd.date_range(start=start_time_hist, periods=len(df_hist), freq='h')

        y_train = df_hist['target']
        X_train = df_hist[['day_of_week', 'month']]

        # Preparar dados futuros
        prediction_length = len(df_future_cov)
        start_time_future = end_time_hist + pd.DateOffset(hours=1)
        df_future_cov.index = pd.date_range(start=start_time_future, periods=len(df_future_cov), freq='h')
        X_future = df_future_cov[['day_of_week', 'month']]

        print("Dados de treino (y):", y_train.shape)
        print("Dados de treino (X):", X_train.shape)
        print("Covariáveis futuras (X):", X_future.shape)

        # Criar o horizonte de previsão
        fh = ForecastingHorizon(X_future.index, is_relative=False)

        # Configurar parâmetros do modelo
        model_config = {
            "context_length": len(y_train),
            "prediction_length": prediction_length
        }

        print("\nInstanciando e treinando o modelo...")
        model = TinyTimeMixerForecaster(
            model_path='ibm-granite/granite-timeseries-ttm-r2',
            config=model_config
        )

        # Treinar o modelo, passando o fh
        model.fit(y_train, X=X_train, fh=fh)

        # Fazer a previsão
        print("Fazendo a previsão...")
        y_pred = model.predict(X=X_future)

        # Visualizar resultados
        print("Plotando resultados...")
        plt.figure(figsize=(12, 6))
        plt.plot(y_train.index, y_train, label='Histórico (Alvo)')
        plt.plot(y_pred.index, y_pred, label='Previsão', color='red')
        plt.title('Previsão TinyTimeMixer com Covariáveis (via sktime)')
        plt.xlabel('Data')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)

        # Salvar o gráfico
        plt.savefig('tinytimemixer_forecast.png')
        print("Gráfico salvo como tinytimemixer_forecast.png")

    except Exception as e:
        print(f"Ocorreu um erro detalhado:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
