"""
IBM Flowstate (granite-timeseries-flowstate-r1): Time-Scale Adjustable Time Series Foundation Model

Flowstate combina State Space Model Encoder com Functional Basis Decoder.
Suporte para diferentes escalas temporais com scale_factor.
"""

import os
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

# Instalar granite-tsfm se necessário
try:
    from tsfm_public import FlowStateForPrediction
except ImportError:
    print("Instalando granite-tsfm...")
    os.system("pip install git+https://github.com/ibm-granite/granite-tsfm.git")
    from tsfm_public import FlowStateForPrediction

class FlowstateForecaster:
    def __init__(self, model_name="ibm-granite/granite-timeseries-flowstate-r1", 
                 device="auto"):
        """
        Inicializa o forecaster Flowstate
        
        Args:
            model_name: Nome do modelo no Hugging Face
            device: Dispositivo de execução
        """
        self.model_name = model_name
        self.device = device
        
        # Configurar device
        if device == "auto":
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Carregando modelo {model_name} no device {self.device}...")
        
        # Carregar modelo
        self.predictor = FlowStateForPrediction.from_pretrained(
            model_name
        ).to(self.device)
        
        print("Flowstate predictor carregado com sucesso!")
    
    def calculate_scale_factor(self, frequency, seasonal_pattern=None):
        """
        Calcula scale_factor baseado na frequência dos dados
        
        Args:
            frequency: Frequência dos dados ('15min', '30min', '1h', '1d', '1w', '1m')
            seasonal_pattern: Padrão sazonal detectado (ex: 24 para daily, 168 para weekly)
            
        Returns:
            scale_factor calculado
        """
        frequency_mapping = {
            '15min': 0.25,
            '30min': 0.5, 
            '1h': 1.0,
            '2h': 2.0,
            '6h': 6.0,
            '12h': 12.0,
            '1d': 'auto',  # Calcula baseado no padrão sazonal
            '1w': 'auto',
            '1m': 'auto'
        }
        
        base_scale = frequency_mapping.get(frequency, 1.0)
        
        if base_scale == 'auto':
            if frequency == '1d':
                if seasonal_pattern == 7:  # Weekly cycle
                    scale_factor = 3.43
                else:
                    scale_factor = 0.0656
            elif frequency == '1w':
                scale_factor = 0.46
            elif frequency == '1m':
                scale_factor = 2.0
            else:
                scale_factor = 1.0
        else:
            scale_factor = base_scale
        
        print(f"Scale factor calculado:")
        print(f"- Frequência: {frequency}")
        print(f"- Padrão sazonal: {seasonal_pattern}")
        print(f"- Scale factor: {scale_factor}")
        
        return scale_factor
    
    def prepare_univariate_data(self, data, target_col, time_col, 
                              series_id_col=None, frequency='1h'):
        """
        Prepara dados univariados para Flowstate
        
        Args:
            data: DataFrame com os dados
            target_col: Nome da coluna alvo
            time_col: Nome da coluna de tempo
            series_id_col: Nome da coluna de ID da série
            frequency: Frequência dos dados
        """
        # Ordenar por tempo
        data = data.sort_values(time_col).reset_index(drop=True)
        
        # Calcular scale factor
        scale_factor = self.calculate_scale_factor(frequency)
        
        # Extrair série
        if series_id_col and series_id_col in data.columns:
            unique_ids = data[series_id_col].unique()
            series_data = data[data[series_id_col] == unique_ids[0]]
            print(f"Usando série: {unique_ids[0]}")
        else:
            series_data = data
        
        # Converter para tensor (context, batch, n_channels)
        time_series = series_data[target_col].values
        n_timesteps = len(time_series)
        
        # Reshape para (context_length, batch_size, n_channels)
        # Por simplicidade, assumir batch=1, n_channels=1
        input_tensor = torch.tensor(
            time_series.reshape(-1, 1, 1), 
            dtype=torch.float32,
            device=self.device
        )
        
        print(f"Dados univariados preparados:")
        print(f"- Shape: {input_tensor.shape}")
        print(f"- Scale factor: {scale_factor}")
        print(f"- Frequência: {frequency}")
        
        return input_tensor, scale_factor
    
    def prepare_multivariate_data(self, data, target_cols, time_col, 
                                series_id_col=None, frequency='1h'):
        """
        Prepara dados multivariados para Flowstate
        
        Args:
            data: DataFrame com os dados
            target_cols: Lista de colunas alvo
            time_col: Nome da coluna de tempo
            series_id_col: Nome da coluna de ID da série
            frequency: Frequência dos dados
        """
        # Ordenar por tempo
        data = data.sort_values(time_col).reset_index(drop=True)
        
        # Calcular scale factor
        scale_factor = self.calculate_scale_factor(frequency)
        
        # Extrair séries
        if series_id_col and series_id_col in data.columns:
            unique_ids = data[series_id_col].unique()
            series_data = data[data[series_id_col] == unique_ids[0]]
            print(f"Usando série: {unique_ids[0]}")
        else:
            series_data = data
        
        # Extrair múltiplas colunas
        series_arrays = []
        for col in target_cols:
            series_arrays.append(series_data[col].values)
        
        # Stack verticalmente e transpor para (context, batch, channels)
        multivariate_data = np.stack(series_arrays, axis=1)  # (time, channels)
        input_tensor = torch.tensor(
            multivariate_data, 
            dtype=torch.float32,
            device=self.device
        ).transpose(0, 1).unsqueeze(1)  # (context, 1, channels)
        
        print(f"Dados multivariados preparados:")
        print(f"- Shape: {input_tensor.shape}")
        print(f"- Targets: {target_cols}")
        print(f"- Scale factor: {scale_factor}")
        
        return input_tensor, scale_factor
    
    def forecast(self, input_tensor, scale_factor, prediction_length=96, batch_first=False):
        """
        Gera forecast usando Flowstate
        
        Args:
            input_tensor: Dados de entrada
            scale_factor: Fator de escala
            prediction_length: Passos para prever
            batch_first: Se batch é a primeira dimensão
            
        Returns:
            Tensor com previsões
        """
        print(f"Gerando forecast:")
        print(f"- Input shape: {input_tensor.shape}")
        print(f"- Scale factor: {scale_factor}")
        print(f"- Prediction length: {prediction_length}")
        
        # Fazer forecast
        forecast = self.predictor(
            input_tensor,
            scale_factor=scale_factor,
            prediction_length=prediction_length,
            batch_first=batch_first
        )
        
        print(f"Forecast shape: {forecast.shape}")
        print("Forecast gerado com sucesso!")
        
        return forecast

def create_time_scale_data(n_timesteps=1000, frequency='1h', n_series=1):
    """
    Cria dados sintéticos para diferentes escalas temporais
    """
    np.random.seed(42)
    data_list = []
    
    # Mapeamento de frequência
    freq_to_timedelta = {
        '15min': '15min', '30min': '30min', '1h': '1h',
        '6h': '6h', '1d': '1d', '1w': '1w', '1m': '1M'
    }
    
    timedelta_str = freq_to_timedelta.get(frequency, '1h')
    dates = pd.date_range('2020-01-01', periods=n_timesteps, freq=timedelta_str)
    
    for series_id in range(n_series):
        # Gerar série com padrões apropriados para a frequência
        if frequency in ['15min', '30min', '1h']:
            # Padrão intradiário
            trend = np.linspace(100, 120, n_timesteps)
            daily_pattern = 15 * np.sin(2 * np.pi * np.arange(n_timesteps) / (24 * (60 if frequency == '15min' else 30)))
            noise = np.random.normal(0, 3, n_timesteps)
            target = trend + daily_pattern + noise
            
        elif frequency in ['6h', '1d']:
            # Padrão diário
            trend = np.linspace(100, 140, n_timesteps)
            daily_pattern = 20 * np.sin(2 * np.pi * np.arange(n_timesteps) / 4)
            noise = np.random.normal(0, 4, n_timesteps)
            target = trend + daily_pattern + noise
            
        elif frequency in ['1w', '1m']:
            # Padrão semanal/mensal
            trend = np.linspace(100, 160, n_timesteps)
            seasonal_pattern = 25 * np.sin(2 * np.pi * np.arange(n_timesteps) / 52)  # Weekly
            noise = np.random.normal(0, 5, n_timesteps)
            target = trend + seasonal_pattern + noise
        
        # Adicionar séries extras para multivariado
        related_metric = 0.7 * target + 0.3 * np.random.normal(100, 10, n_timesteps)
        derived_metric = target * 0.1 + np.random.normal(0, 1, n_timesteps)
        
        for i, date in enumerate(dates):
            data_list.append({
                'timestamp': date,
                'primary_metric': target[i],
                'related_metric': related_metric[i],
                'derived_metric': derived_metric[i],
                'series_id': f'series_{series_id}'
            })
    
    return pd.DataFrame(data_list)

def example_univariate_different_frequencies():
    """
    Exemplo de forecasting univariado em diferentes frequências
    """
    print("=== Exemplo 1: Univariado em Diferentes Frequências ===")
    
    frequencies = ['15min', '1h', '1d']
    results = {}
    
    for freq in frequencies:
        print(f"\\n--- Frequência: {freq} ---")
        
        # Criar dados
        n_samples = 500 if freq == '15min' else 300
        data = create_time_scale_data(n_timesteps=n_samples, frequency=freq, n_series=1)
        
        print(f"Dados: {data.shape}")
        print(f"Período: {data['timestamp'].min()} a {data['timestamp'].max()}")
        
        # Inicializar forecaster
        forecaster = FlowstateForecaster(
            model_name="ibm-granite/granite-timeseries-flowstate-r1"
        )
        
        # Preparar dados
        input_tensor, scale_factor = forecaster.prepare_univariate_data(
            data,
            target_col='primary_metric',
            time_col='timestamp',
            series_id_col='series_id',
            frequency=freq
        )
        
        # Dividir dados
        train_size = int(0.85 * len(input_tensor))
        train_tensor = input_tensor[:train_size]
        test_size = min(96, len(input_tensor) - train_size)  # Máximo 96 para avaliação
        
        print(f"\\nConfiguração:")
        print(f"- Treino: {len(train_tensor)} timesteps")
        print(f"- Teste: {test_size} timesteps")
        print(f"- Scale factor: {scale_factor}")
        
        # Forecast
        forecast = forecaster.forecast(
            train_tensor,
            scale_factor=scale_factor,
            prediction_length=test_size
        )
        
        # Extrair previsões
        # forecast shape: [batch, quantiles, forecast_length, channels]
        predictions = forecast[0, 4, :, 0].cpu().numpy()  # Mediana (quantile 0.5)
        
        # Valores reais para avaliação
        actual_values = input_tensor[train_size:train_size+test_size, 0, 0].cpu().numpy()
        
        # Métricas
        mae = np.mean(np.abs(predictions - actual_values))
        rmse = np.sqrt(np.mean((predictions - actual_values)**2))
        
        results[freq] = {
            'predictions': predictions,
            'actual': actual_values,
            'mae': mae,
            'rmse': rmse,
            'scale_factor': scale_factor
        }
        
        print(f"\\nResultados:")
        print(f"- MAE: {mae:.2f}")
        print(f"- RMSE: {rmse:.2f}")
        print(f"- Previsões (primeiros 5): {predictions[:5]}")
        print(f"- Reais (primeiros 5): {actual_values[:5]}")
    
    return results

def example_multivariate_forecasting():
    """
    Exemplo de forecasting multivariado
    """
    print("\\n=== Exemplo 2: Forecasting Multivariado ===")
    
    # Criar dados hourly com múltiplas métricas
    data = create_time_scale_data(n_timesteps=400, frequency='1h', n_series=2)
    
    print("Dados multivariados:")
    print(data.head())
    print(f"Shape: {data.shape}")
    print(f"Colunas: {list(data.columns)}")
    
    # Inicializar forecaster
    forecaster = FlowstateForecaster(
        model_name="ibm-granite/granite-timeseries-flowstate-r1"
    )
    
    # Preparar dados multivariados
    target_cols = ['primary_metric', 'related_metric', 'derived_metric']
    
    data_dict = {}
    for series_id in data['series_id'].unique():
        series_data = data[data['series_id'] == series_id]
        
        input_tensor, scale_factor = forecaster.prepare_multivariate_data(
            series_data,
            target_cols=target_cols,
            time_col='timestamp',
            series_id_col=None,
            frequency='1h'
        )
        
        data_dict[series_id] = {
            'tensor': input_tensor,
            'scale_factor': scale_factor
        }
    
    # Processar cada série
    all_results = {}
    
    for series_id, series_data in data_dict.items():
        print(f"\\nProcessando {series_id}:")
        print(f"- Input shape: {series_data['tensor'].shape}")
        
        # Dividir
        train_size = int(0.8 * len(series_data['tensor']))
        train_tensor = series_data['tensor'][:train_size]
        test_size = 48
        
        # Forecast
        forecast = forecaster.forecast(
            train_tensor,
            scale_factor=series_data['scale_factor'],
            prediction_length=test_size
        )
        
        # Extrair previsões para cada métrica
        # [batch, quantiles, forecast_length, channels]
        predictions = {}
        actual_values = {}
        
        for i, col in enumerate(target_cols):
            pred = forecast[0, 4, :, i].cpu().numpy()  # Mediana
            actual = series_data['tensor'][train_size:train_size+test_size, 0, i].cpu().numpy()
            
            predictions[col] = pred
            actual_values[col] = actual
        
        # Métricas
        series_results = {}
        for col in target_cols:
            mae = np.mean(np.abs(predictions[col] - actual_values[col]))
            rmse = np.sqrt(np.mean((predictions[col] - actual_values[col])**2))
            
            series_results[col] = {
                'predictions': predictions[col],
                'actual': actual_values[col],
                'mae': mae,
                'rmse': rmse
            }
        
        all_results[series_id] = series_results
        
        print(f"Resultados para {series_id}:")
        for col, metrics in series_results.items():
            print(f"  {col}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
    
    return all_results

def example_scale_factor_sensitivity():
    """
    Exemplo mostrando sensibilidade ao scale_factor
    """
    print("\\n=== Exemplo 3: Sensibilidade ao Scale Factor ===")
    
    # Criar dados diários
    data = create_time_scale_data(n_timesteps=200, frequency='1d', n_series=1)
    
    print("Dados diários:")
    print(data.head())
    print(f"Shape: {data.shape}")
    
    # Inicializar forecaster
    forecaster = FlowstateForecaster(
        model_name="ibm-granite/granite-timeseries-flowstate-r1"
    )
    
    # Preparar dados
    input_tensor, default_scale = forecaster.prepare_univariate_data(
        data,
        target_col='primary_metric',
        time_col='timestamp',
        series_id_col='series_id',
        frequency='1d'
    )
    
    # Dividir
    train_size = int(0.8 * len(input_tensor))
    train_tensor = input_tensor[:train_size]
    test_size = 30
    
    # Testar diferentes scale factors
    scale_factors = [0.0656, 3.43, 1.0, 2.0]  # Padrão, weekly cycle, 1x, 2x
    scale_names = ['Padrão (0.0656)', 'Weekly cycle (3.43)', '1x (1.0)', '2x (2.0)']
    
    results = {}
    
    for scale_factor, scale_name in zip(scale_factors, scale_names):
        print(f"\\nTestando scale factor: {scale_name}")
        
        # Forecast
        forecast = forecaster.forecast(
            train_tensor,
            scale_factor=scale_factor,
            prediction_length=test_size
        )
        
        # Extrair previsões
        predictions = forecast[0, 4, :, 0].cpu().numpy()
        actual_values = input_tensor[train_size:train_size+test_size, 0, 0].cpu().numpy()
        
        # Métricas
        mae = np.mean(np.abs(predictions - actual_values))
        rmse = np.sqrt(np.mean((predictions - actual_values)**2))
        
        results[scale_name] = {
            'predictions': predictions,
            'actual': actual_values,
            'mae': mae,
            'rmse': rmse
        }
        
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
    
    # Encontrar melhor scale factor
    best_scale = min(results.keys(), key=lambda x: results[x]['mae'])
    print(f"\\nMelhor scale factor: {best_scale} (MAE: {results[best_scale]['mae']:.2f})")
    
    return results

def example_long_term_forecasting():
    """
    Exemplo de forecasting de longo prazo
    """
    print("\\n=== Exemplo 4: Forecasting de Longo Prazo ===")
    
    # Criar dados com muitos pontos
    data = create_time_scale_data(n_timesteps=800, frequency='1h', n_series=1)
    
    print("Dados para longo prazo:")
    print(f"Shape: {data.shape}")
    
    # Inicializar forecaster
    forecaster = FlowstateForecaster(
        model_name="ibm-granite/granite-timeseries-flowstate-r1"
    )
    
    # Preparar dados
    input_tensor, scale_factor = forecaster.prepare_univariate_data(
        data,
        target_col='primary_metric',
        time_col='timestamp',
        series_id_col='series_id',
        frequency='1h'
    )
    
    # Dividir com mais contexto
    train_size = int(0.9 * len(input_tensor))
    train_tensor = input_tensor[:train_size]
    
    print(f"\\nConfiguração para longo prazo:")
    print(f"- Contexto: {len(train_tensor)} timesteps")
    print(f"- Scale factor: {scale_factor}")
    
    # Forecasts de diferentes horizontes
    horizons = [24, 48, 96, 192]  # 1d, 2d, 4d, 8d
    results = {}
    
    for horizon in horizons:
        print(f"\\nForecast para {horizon} timesteps:")
        
        # Forecast
        forecast = forecaster.forecast(
            train_tensor,
            scale_factor=scale_factor,
            prediction_length=horizon
        )
        
        # Extrair previsões
        predictions = forecast[0, 4, :, 0].cpu().numpy()
        
        # Análise de qualidade
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        trend = np.polyfit(range(horizon), predictions, 1)[0]  # Tendência linear
        
        results[horizon] = {
            'predictions': predictions,
            'mean': mean_pred,
            'std': std_pred,
            'trend': trend
        }
        
        print(f"  - Média: {mean_pred:.2f}")
        print(f"  - Desvio: {std_pred:.2f}")
        print(f"  - Tendência: {trend:.4f}")
        print(f"  - Primeiras 5: {predictions[:5]}")
    
    print("\\nNota: Flowstate recomenda forecast de no máximo 30 sazonalidades")
    print("Para este caso (hourly), máximo recomendado seria ~720 timesteps (30 dias)")
    
    return results

if __name__ == "__main__":
    print("=== IBM Flowstate Time Series Forecasting ===\\n")
    
    try:
        # Exemplo 1: Diferentes frequências
        results1 = example_univariate_different_frequencies()
        
        # Exemplo 2: Multivariado
        results2 = example_multivariate_forecasting()
        
        # Exemplo 3: Sensibilidade scale factor
        results3 = example_scale_factor_sensitivity()
        
        # Exemplo 4: Longo prazo
        results4 = example_long_term_forecasting()
        
        print("\\n=== Resumo ===")
        print("✓ Exemplo 1: Univariado em diferentes frequências")
        print("✓ Exemplo 2: Forecasting multivariado")
        print("✓ Exemplo 3: Sensibilidade ao scale factor")
        print("✓ Exemplo 4: Forecasting de longo prazo")
        print("\\nFlowstate é ideal para:")
        print("- Séries em diferentes escalas temporais")
        print("- Dados com patterns sazonais")
        print("- Forecasts com ajuste automático de escala")
        print("- Modelos compactos (< 10M parâmetros)")
        
    except Exception as e:
        print(f"Erro durante execução: {e}")
        print("\\nPara usar Flowstate, certifique-se de que granite-tsfm está instalado:")
        print("pip install git+https://github.com/ibm-granite/granite-tsfm.git")