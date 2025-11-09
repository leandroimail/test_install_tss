"""
Amazon Chronos-2: Zero-Shot Time Series Forecasting with Dynamic Covariates

Chronos-2 é um modelo de foundation de 120M parâmetros para forecasting zero-shot.
Suporte nativo para covariáveis dinâmicas através dos parâmetros future_df e context_df.
"""

import pandas as pd
import numpy as np
from chronos import Chronos2Pipeline
import torch
import warnings
warnings.filterwarnings('ignore')

class Chronos2Forecaster:
    def __init__(self, model_name="amazon/chronos-2", device="auto"):
        """
        Inicializa o forecaster Chronos-2
        
        Args:
            model_name: Nome do modelo no Hugging Face
            device: Disposito de execução ('auto', 'cuda', 'cpu')
        """
        print(f"Carregando modelo {model_name}...")
        self.pipeline = Chronos2Pipeline.from_pretrained(
            model_name, 
            device_map=device
        )
        print("Modelo carregado com sucesso!")
    
    def prepare_data(self, data, id_col, timestamp_col, target_col, 
                     past_covariates=None, future_covariates=None):
        """
        Prepara os dados para o modelo Chronos-2
        
        Args:
            data: DataFrame com os dados
            id_col: Nome da coluna de identificação
            timestamp_col: Nome da coluna de timestamp
            target_col: Nome da coluna alvo
            past_covariates: Lista de nomes das covariáveis passadas
            future_covariates: Lista de nomes das covariáveis futuras
        """
        # Dados de contexto (históricos) incluem target + covariáveis passadas
        context_cols = [timestamp_col, target_col]
        if past_covariates:
            context_cols.extend(past_covariates)
        
        self.context_df = data[context_cols + [id_col]].copy()
        self.context_df.rename(columns={timestamp_col: 'timestamp'}, inplace=True)
        self.context_df.rename(columns={target_col: 'target'}, inplace=True)
        
        # Dados futuros (covariáveis conhecidas no futuro)
        if future_covariates:
            future_cols = [timestamp_col] + future_covariates + [id_col]
            self.future_df = data[future_cols].copy()
            self.future_df.rename(columns={timestamp_col: 'timestamp'}, inplace=True)
        else:
            self.future_df = None
        
        self.id_column = id_col
        self.timestamp_column = 'timestamp'
        self.target = 'target'
        
        print(f"Dados preparados:")
        print(f"- Contexto: {self.context_df.shape}")
        if self.future_df is not None:
            print(f"- Covariáveis futuras: {self.future_df.shape}")
    
    def forecast(self, prediction_length=24, quantile_levels=[0.1, 0.5, 0.9]):
        """
        Gera forecast usando Chronos-2 com covariáveis dinâmicas
        
        Args:
            prediction_length: Número de passos para prever
            quantile_levels: Níveis de quantil para previsão probabilística
            
        Returns:
            DataFrame com as previsões
        """
        print(f"Gerando forecast para {prediction_length} passos...")
        
        # Previsão
        pred_df = self.pipeline.predict_df(
            context_df=self.context_df,
            future_df=self.future_df,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
            id_column=self.id_column,
            timestamp_column=self.timestamp_column,
            target=self.target,
        )
        
        print("Forecast gerado com sucesso!")
        return pred_df

def example_electricity_forecasting():
    """
    Exemplo de uso do Chronos-2 com dados de preço de electricidade
    """
    # Instalar dependências se necessário
    # !pip install chronos-forecasting pandas numpy
    
    # Inicializar o forecaster
    forecaster = Chronos2Forecaster()
    
    # Carregar dados de exemplo (preço de electricidade)
    context_df = pd.read_parquet(
        "https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/train.parquet"
    )
    test_df = pd.read_parquet(
        "https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/test.parquet"
    )
    
    # Separar dados de contexto e covariáveis futuras
    # Assumindo que 'target' é a coluna alvo e não há covariáveis
    print("Dados carregados:")
    print(f"Contexto: {context_df.shape}")
    print(f"Teste: {test_df.shape}")
    print(f"Colunas: {list(context_df.columns)}")
    
    # Preparar dados (sem covariáveis por simplicidade)
    forecaster.prepare_data(
        data=context_df,
        id_col='id',
        timestamp_col='timestamp', 
        target_col='target'
    )
    
    # Gerar forecast
    prediction_length = 24  # 24 horas
    forecast = forecaster.forecast(
        prediction_length=prediction_length,
        quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9]
    )
    
    print(f"\nForecast shape: {forecast.shape}")
    print(f"Primeiras 5 linhas:")
    print(forecast.head())
    
    return forecast

def example_with_dynamic_covariates():
    """
    Exemplo demonstrando como usar covariáveis dinâmicas com Chronos-2
    """
    # Criar dados sintéticos com covariáveis dinâmicas
    np.random.seed(42)
    n_timesteps = 1000
    n_series = 3
    
    # Gerar timestamps
    timestamps = pd.date_range('2020-01-01', periods=n_timesteps, freq='H')
    
    # Dados sintéticos
    data_list = []
    for series_id in range(n_series):
        # Série temporal principal (target)
        trend = np.linspace(100, 120, n_timesteps)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_timesteps) / 24)
        noise = np.random.normal(0, 2, n_timesteps)
        target = trend + seasonal + noise
        
        # Covariável dinâmica 1: Temperatura
        temp_base = 20 + 5 * np.sin(2 * np.pi * np.arange(n_timesteps) / (24 * 365))
        temp_seasonal = 3 * np.sin(2 * np.pi * np.arange(n_timesteps) / 24)
        temperature = temp_base + temp_seasonal + np.random.normal(0, 1, n_timesteps)
        
        # Covariável dinâmica 2: Preço de energia
        price = 50 + 10 * np.sin(2 * np.pi * np.arange(n_timesteps) / 168) + np.random.normal(0, 2, n_timesteps)
        
        # Covariável categórica: Dia da semana (0=domingo, 6=sábado)
        weekday = (timestamps.dayofweek % 7).astype(int)
        
        for i, ts in enumerate(timestamps):
            data_list.append({
                'id': f'series_{series_id}',
                'timestamp': ts,
                'target': target[i],
                'temperature': temperature[i],
                'energy_price': price[i],
                'weekday': weekday[i]
            })
    
    data = pd.DataFrame(data_list)
    
    print("Dados sintéticos criados:")
    print(data.head())
    print(f"Shape: {data.shape}")
    print(f"Colunas: {list(data.columns)}")
    
    # Inicializar e preparar forecaster
    forecaster = Chronos2Forecaster()
    
    # Dividir dados
    train_size = int(0.8 * len(data) / n_series) * n_series
    train_data = data[data.index % n_series < train_size // n_series]
    # Para o exemplo, usaremos os últimos pontos como contexto
    context_data = data.iloc[-800:]  # Últimos 800 pontos para contexto
    
    # Preparar com covariáveis dinâmicas
    forecaster.prepare_data(
        data=context_data,
        id_col='id',
        timestamp_col='timestamp',
        target_col='target',
        past_covariates=['temperature', 'energy_price', 'weekday']
    )
    
    # Gerar forecast
    prediction_length = 48  # 48 horas
    forecast = forecaster.forecast(
        prediction_length=prediction_length,
        quantile_levels=[0.1, 0.5, 0.9]
    )
    
    print(f"\nForecast com covariáveis:")
    print(f"Shape: {forecast.shape}")
    print(forecast.head())
    
    return forecast

if __name__ == "__main__":
    print("=== Amazon Chronos-2 Time Series Forecasting ===\n")
    
    # Exemplo 1: Dados de electricidade
    print("Exemplo 1: Forecasting de preços de electricidade")
    forecast1 = example_electricity_forecasting()
    
    print("\n" + "="*50 + "\n")
    
    # Exemplo 2: Com covariáveis dinâmicas
    print("Exemplo 2: Forecasting com covariáveis dinâmicas")
    forecast2 = example_with_dynamic_covariates()
    
    print("\nExemplos concluídos!")