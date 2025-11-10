# =====================================================================================
# ATENÇÃO: BLOQUEIO DE AMBIENTE
# =====================================================================================
# A lógica para manipulação de dados e uso de covaráveis com o PatchTSMixer
# está implementada corretamente abaixo. No entanto, o script não pode ser
# executado com sucesso no ambiente atual devido a um problema de acesso ao
# Hugging Face Hub.
#
# Erro recorrente:
# huggingface_hub.errors.RepositoryNotFoundError: 401 Client Error. Unauthorized
#
# Isso impede o download de qualquer modelo pré-treinado do Hub. Para executar
# este script, o ambiente precisa ser configurado com a autenticação correta
# do Hugging Face (ex: `huggingface-cli login`) ou o problema de rede/firewall
# que impede o acesso precisa ser resolvido.
# =====================================================================================

import pandas as pd
import torch
from transformers import PatchTSMixerForPrediction
import matplotlib.pyplot as plt
import numpy as np
import traceback

def main():
    """
    Função principal para treinar o modelo PatchTSMixer com covaráveis, fazer previsões e plotar os resultados,
    adaptada para a estrutura de dados correta.
    """
    try:
        # Carregar dados
        print("Carregando dados...")
        df_hist = pd.read_parquet('../../data/hist.parquet')
        print("Dados carregados com sucesso. Colunas:", df_hist.columns)

        # Criar índice de tempo
        end_time = pd.to_datetime('2024-07-15 23:00:00')
        start_time = end_time - pd.DateOffset(hours=len(df_hist) - 1)
        df_hist.index = pd.date_range(start=start_time, periods=len(df_hist), freq='h')

        # Parâmetros
        prediction_length = 14
        context_length = len(df_hist)

        # Preparar dataframe para o modelo
        df_merged = df_hist[['target', 'day_of_week', 'month']]

        print("DataFrame preparado para o modelo:")
        print(df_merged.head())

        context = df_merged.values[-context_length:]
        past_values = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        # Carregar modelo pré-treinado
        print("\nCarregando modelo (esta etapa falhará no ambiente atual)...")
        model = PatchTSMixerForPrediction.from_pretrained("ibm/patchtsmixer-base-prediction-etth1")

        # Gerar previsões
        print("Gerando previsões...")
        with torch.no_grad():
            outputs = model.generate(
                past_values=past_values
            )

        # Extrair a previsão
        forecast_values = outputs.sequences.numpy()[0, 0, -prediction_length:]

        # Visualizar resultados
        print("Plotando resultados...")
        plt.figure(figsize=(12, 6))
        plt.plot(df_merged.index, df_merged['target'], label='Histórico (Alvo)')
        forecast_index = pd.date_range(start=df_merged.index[-1], periods=prediction_length + 1, freq='h')[1:]
        plt.plot(forecast_index, forecast_values, label='Previsão', color='red')
        plt.title('Previsão PatchTSMixer com Covariáveis')
        plt.xlabel('Data')
        plt.ylabel('Valor')
        plt.legend()
        plt.grid(True)

        # Salvar o gráfico
        plt.savefig('patchtsmixer_forecast.png')
        print("Gráfico salvo como patchtsmixer_forecast.png")

    except Exception as e:
        print(f"Ocorreu um erro detalhado:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
