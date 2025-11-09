"""
Google TimesFM-2.5: Time Series Foundation Model with Dynamic Covariates

TimesFM-2.5 é um modelo de foundation para forecasting com 200M parâmetros.
Suporte para covariáveis através da instalação XReg (uv pip install -e .[xreg]).
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

# Instalar timesfm se necessário
try:
    import timesfm
    print("TimesFM já instalado")
except ImportError:
    print("Instalando TimesFM...")
    # Método 1: Clone do repositório
    os.system("git clone https://github.com/google-research/timesfm.git")
    os.chdir("timesfm")
    os.system("pip install -e .")
    sys.path.append("/workspace/timesfm")
    import timesfm

class TimesFM2Forecaster:
    def __init__(self, model_variant="google/timesfm-2.5-200m-pytorch", 
                 context_length=1024, horizon=256, device="auto"):
        """
        Inicializa o forecaster TimesFM-2.5
        
        Args:
            model_variant: Variante do modelo
            context_length: Comprimento do contexto
            horizon: Horizonte de previsão
            device: Dispositivo de execução
        """
        self.model_variant = model_variant
        self.context_length = context_length
        self.horizon = horizon
        self.device = device
        
        # Configurar precisão matemática
        torch.set_float32_matmul_precision("high")
        
        print(f"Carregando modelo {model_variant}...")
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            model_variant,
            torch_compile=True if device == "cuda" else False
        )
        
        # Configurar modelo
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=context_length,
                max_horizon=horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        print("Modelo TimesFM-2.5 carregado e configurado!")
    
    def prepare_univariate_data(self, data, target_col, time_col, series_id_col=None):
        """
        Prepara dados univariados para TimesFM
        
        Args:
            data: DataFrame com os dados
            target_col: Nome da coluna alvo
            time_col: Nome da coluna de tempo
            series_id_col: Nome da coluna de ID da série
        """
        # Ordenar por tempo
        data = data.sort_values(time_col).reset_index(drop=True)
        
        # Extrair série temporal
        if series_id_col and series_id_col in data.columns:
            # Se há múltiplas séries, pegar a primeira
            unique_series = data[series_id_col].unique()
            series_data = data[data[series_id_col] == unique_series[0]]
            print(f"Usando série: {unique_series[0]}")
        else:
            series_data = data
        
        # Extrair valores
        time_series = series_data[target_col].values
        timestamps = pd.to_datetime(series_data[time_col])
        
        print(f"Dados univariados preparados:")
        print(f"- Série shape: {time_series.shape}")
        print(f"- Período: {timestamps.min()} a {timestamps.max()}")
        print(f"- Valores: {time_series[:5]}...")
        
        return time_series, timestamps
    
    def prepare_multivariate_data(self, data, target_cols, time_col, series_id_col=None):
        """
        Prepara dados multivariados para TimesFM
        
        Args:
            data: DataFrame com os dados
            target_cols: Lista de colunas alvo
            time_col: Nome da coluna de tempo
            series_id_col: Nome da coluna de ID da série
        """
        # Ordenar por tempo
        data = data.sort_values(time_col).reset_index(drop=True)
        
        # Extrair múltiplas séries
        if series_id_col and series_id_col in data.columns:
            unique_series = data[series_id_col].unique()
            if len(unique_series) > 1:
                # Para múltiplas séries, preparar lista de arrays
                series_list = []
                for series_id in unique_series:
                    series_data = data[data[series_id_col] == series_id]
                    series_array = series_data[target_cols].values
                    series_list.append(series_array)
                series_arrays = series_list
                print(f"Múltiplas séries encontradas: {len(series_arrays)}")
            else:
                # Única série
                series_data = data[data[series_id_col] == unique_series[0]]
                series_arrays = [series_data[target_cols].values]
        else:
            # Sem ID, assumir única série
            series_arrays = [data[target_cols].values]
        
        # Verificar shapes
        for i, arr in enumerate(series_arrays):
            print(f"- Série {i+1} shape: {arr.shape}")
        
        return series_arrays
    
    def forecast_univariate(self, time_series, horizon=24, freq=None):
        """
        Forecast univariado usando TimesFM
        
        Args:
            time_series: Série temporal
            horizon: Passos para prever
            freq: Frequência dos dados
            
        Returns:
            Tupla (ponto_forecast, quantil_forecast)
        """
        print(f"Gerando forecast univariado:")
        print(f"- Série shape: {time_series.shape}")
        print(f"- Horizonte: {horizon}")
        
        # Converter para numpy se necessário
        if isinstance(time_series, pd.Series):
            time_series = time_series.values
        
        # Fazer forecast
        point_forecast, quantile_forecast = self.model.forecast(
            horizon=horizon,
            inputs=[time_series],
        )
        
        print(f"- Ponto forecast shape: {point_forecast.shape}")
        print(f"- Quantil forecast shape: {quantile_forecast.shape}")
        
        return point_forecast, quantile_forecast
    
    def forecast_multivariate(self, series_arrays, horizon=24, freq=None):
        """
        Forecast multivariado usando TimesFM
        
        Args:
            series_arrays: Lista de arrays multivariados
            horizon: Passos para prever
            freq: Frequência dos dados
            
        Returns:
            Tupla (ponto_forecast, quantil_forecast)
        """
        print(f"Gerando forecast multivariado:")
        print(f"- Número de séries: {len(series_arrays)}")
        print(f"- Horizonte: {horizon}")
        
        # Fazer forecast para cada série
        point_forecasts = []
        quantile_forecasts = []
        
        for i, series in enumerate(series_arrays):
            print(f"  - Série {i+1}: {series.shape}")
            
            point_forecast, quantile_forecast = self.model.forecast(
                horizon=horizon,
                inputs=[series],
            )
            
            point_forecasts.append(point_forecast)
            quantile_forecasts.append(quantile_forecast)
        
        return point_forecasts, quantile_forecasts

def create_sample_data(n_timesteps=500, n_series=1, n_features=3):
    """
    Cria dados sintéticos para demonstração
    """
    np.random.seed(42)
    data_list = []
    
    for series_id in range(n_series):
        # Gerar timestamps
        dates = pd.date_range('2020-01-01', periods=n_timesteps, freq='D')
        
        # Série principal (target)
        trend = np.linspace(100, 150, n_timesteps)
        seasonal = 20 * np.sin(2 * np.pi * np.arange(n_timesteps) / 365.25)
        noise = np.random.normal(0, 3, n_timesteps)
        target = trend + seasonal + noise
        
        # Covariáveis dinâmicas
        # Preço (influencia negativa na demanda)
        price = 50 + 5 * np.sin(2 * np.pi * np.arange(n_timesteps) / 30) + np.random.normal(0, 1, n_timesteps)
        
        # Temperatura (sazonalidade)
        temp_base = 20 + 10 * np.sin(2 * np.pi * np.arange(n_timesteps) / 365.25)
        temperature = temp_base + np.random.normal(0, 2, n_timesteps)
        
        # Promoção (binária)
        promotion = np.random.binomial(1, 0.2, n_timesteps)
        
        # Evento especial (catastrófico)
        event = np.random.binomial(1, 0.05, n_timesteps)
        
        for i, date in enumerate(dates):
            data_list.append({
                'date': date,
                'demand': target[i],
                'price': price[i],
                'temperature': temperature[i],
                'promotion': promotion[i],
                'event': event[i],
                'series_id': f'series_{series_id}'
            })
    
    return pd.DataFrame(data_list)

def example_univariate_forecasting():
    """
    Exemplo de forecasting univariado com TimesFM
    """
    print("=== Exemplo 1: Forecasting Univariado ===")
    
    # Criar dados sintéticos
    data = create_sample_data(n_timesteps=300)
    
    print("Dados criados:")
    print(data.head())
    print(f"Shape: {data.shape}")
    
    # Inicializar forecaster
    forecaster = TimesFM2Forecaster(
        model_variant="google/timesfm-2.5-200m-pytorch",
        context_length=512,
        horizon=48
    )
    
    # Preparar dados
    time_series, timestamps = forecaster.prepare_univariate_data(
        data,
        target_col='demand',
        time_col='date',
        series_id_col='series_id'
    )
    
    # Dividir dados
    train_size = int(0.8 * len(time_series))
    train_series = time_series[:train_size]
    test_series = time_series[train_size:]
    
    print(f"\\nDivisão dos dados:")
    print(f"- Treino: {len(train_series)} pontos")
    print(f"- Teste: {len(test_series)} pontos")
    
    # Forecast
    point_forecast, quantile_forecast = forecaster.forecast_univariate(
        train_series,
        horizon=len(test_series)
    )
    
    # Avaliar
    mae = np.mean(np.abs(point_forecast[0] - test_series))
    rmse = np.sqrt(np.mean((point_forecast[0] - test_series)**2))
    
    print(f"\\nResultados:")
    print(f"- MAE: {mae:.2f}")
    print(f"- RMSE: {rmse:.2f}")
    print(f"- Ponto forecast: {point_forecast[0][:5]}")
    print(f"- Valores reais: {test_series[:5]}")
    
    return point_forecast, quantile_forecast, test_series

def example_multivariate_forecasting():
    """
    Exemplo de forecasting multivariado
    """
    print("\\n=== Exemplo 2: Forecasting Multivariado ===")
    
    # Criar dados com múltiplas séries
    data = create_sample_data(n_timesteps=200, n_series=2)
    
    print("Dados multivariados:")
    print(data.head())
    print(f"Shape: {data.shape}")
    print(f"Séries: {data['series_id'].unique()}")
    
    # Inicializar forecaster
    forecaster = TimesFM2Forecaster(
        model_variant="google/timesfm-2.5-200m-pytorch",
        context_length=256,
        horizon=30
    )
    
    # Preparar dados multivariados
    series_arrays = forecaster.prepare_multivariate_data(
        data,
        target_cols=['demand'],
        time_col='date',
        series_id_col='series_id'
    )
    
    # Dividir cada série
    train_arrays = []
    test_arrays = []
    
    for series in series_arrays:
        train_size = int(0.8 * len(series))
        train_arrays.append(series[:train_size])
        test_arrays.append(series[train_size:])
    
    # Forecast para cada série
    all_results = {}
    
    for i, (train_series, test_series) in enumerate(zip(train_arrays, test_arrays)):
        print(f"\\nProcessando série {i+1}:")
        print(f"- Treino: {train_series.shape}")
        print(f"- Teste: {test_series.shape}")
        
        # Fazer forecast
        point_forecast, quantile_forecast = forecaster.forecast_univariate(
            train_series.flatten(),
            horizon=len(test_series)
        )
        
        # Avaliar
        mae = np.mean(np.abs(point_forecast[0] - test_series.flatten()))
        rmse = np.sqrt(np.mean((point_forecast[0] - test_series.flatten())**2))
        
        all_results[f'series_{i+1}'] = {
            'point_forecast': point_forecast[0],
            'actual': test_series.flatten(),
            'mae': mae,
            'rmse': rmse
        }
        
        print(f"- MAE: {mae:.2f}")
        print(f"- RMSE: {rmse:.2f}")
    
    return all_results

def example_with_covariates_simulation():
    """
    Exemplo simulando uso de covariáveis (TimesFM com XReg)
    Nota: XReg requer instalação especial que pode não estar disponível
    """
    print("\\n=== Exemplo 3: Simulação com Covariáveis ===")
    print("Nota: TimesFM XReg (covariáveis) requer instalação especial")
    print("Demonstração da estrutura que seria usada com XReg")
    
    # Criar dados com múltiplas features
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=400, freq='D')
    
    # Série principal
    trend = np.linspace(100, 140, 400)
    seasonal = 15 * np.sin(2 * np.pi * np.arange(400) / 365.25)
    base_demand = trend + seasonal
    
    # Simular efeito de covariáveis
    price_effect = -0.5  # Preço maior reduz demanda
    temp_effect = 0.3    # Temperatura aumenta demanda (temperatura confort)
    promo_effect = 20    # Promoção aumenta demanda
    
    # Covariáveis
    price = 50 + 3 * np.sin(2 * np.pi * np.arange(400) / 30) + np.random.normal(0, 1, 400)
    temperature = 20 + 8 * np.sin(2 * np.pi * np.arange(400) / 365.25) + np.random.normal(0, 2, 400)
    promotion = np.random.binomial(1, 0.15, 400)
    
    # Calcular demanda com efeito das covariáveis
    demand = (base_demand + 
              price_effect * (price - 50) + 
              temp_effect * (temperature - 20) + 
              promo_effect * promotion + 
              np.random.normal(0, 2, 400))
    
    data = pd.DataFrame({
        'date': dates,
        'demand': demand,
        'price': price,
        'temperature': temperature,
        'promotion': promotion
    })
    
    print("Dados com covariáveis (simulação):")
    print(data.head())
    print(f"\\nCorrelations:")
    print(f"- Demand vs Price: {data['demand'].corr(data['price']):.3f}")
    print(f"- Demand vs Temperature: {data['demand'].corr(data['temperature']):.3f}")
    print(f"- Demand vs Promotion: {data['demand'].corr(data['promotion']):.3f}")
    
    # Inicializar forecaster
    forecaster = TimesFM2Forecaster(
        model_variant="google/timesfm-2.5-200m-pytorch",
        context_length=256,
        horizon=60
    )
    
    # Preparar e forecast
    time_series, timestamps = forecaster.prepare_univariate_data(
        data,
        target_col='demand',
        time_col='date'
    )
    
    # Dividir
    train_size = int(0.85 * len(time_series))
    train_series = time_series[:train_size]
    test_data = data.iloc[train_size:]
    
    print(f"\\nConfiguração:")
    print(f"- Treino: {len(train_series)} pontos")
    print(f"- Teste: {len(test_data)} pontos")
    
    # Forecast
    point_forecast, quantile_forecast = forecaster.forecast_univariate(
        train_series,
        horizon=len(test_data)
    )
    
    # Avaliar
    actual_demand = test_data['demand'].values
    mae = np.mean(np.abs(point_forecast[0] - actual_demand))
    rmse = np.sqrt(np.mean((point_forecast[0] - actual_demand)**2))
    
    print(f"\\nResultados:")
    print(f"- MAE: {mae:.2f}")
    print(f"- RMSE: {rmse:.2f}")
    
    # Análise de erro por condição
    mae_normal = np.mean(np.abs(point_forecast[0][test_data['promotion'] == 0] - 
                                actual_demand[test_data['promotion'] == 0]))
    mae_promo = np.mean(np.abs(point_forecast[0][test_data['promotion'] == 1] - 
                               actual_demand[test_data['promotion'] == 1]))
    
    print(f"\\nAnálise por condição:")
    print(f"- MAE (sem promoção): {mae_normal:.2f}")
    print(f"- MAE (com promoção): {mae_promo:.2f}")
    
    return point_forecast, quantile_forecast, test_data

if __name__ == "__main__":
    print("=== Google TimesFM-2.5 Time Series Forecasting ===\\n")
    
    try:
        # Exemplo 1: Univariado
        pred1, quant1, actual1 = example_univariate_forecasting()
        
        # Exemplo 2: Multivariado
        results2 = example_multivariate_forecasting()
        
        # Exemplo 3: Simulação com covariáveis
        pred3, quant3, test3 = example_with_covariates_simulation()
        
        print("\\n=== Resumo ===")
        print("✓ Exemplo 1: Forecasting univariado concluído")
        print("✓ Exemplo 2: Forecasting multivariado concluído") 
        print("✓ Exemplo 3: Simulação com covariáveis concluída")
        print("\\nNota: Para usar covariáveis reais, instale TimesFM com XReg:")
        print("cd timesfm && uv pip install -e .[xreg]")
        
    except Exception as e:
        print(f"Erro durante execução: {e}")
        print("\\nPara uso completo, certifique-se de que TimesFM está corretamente instalado:")
        print("git clone https://github.com/google-research/timesfm.git")
        print("cd timesfm && pip install -e .")