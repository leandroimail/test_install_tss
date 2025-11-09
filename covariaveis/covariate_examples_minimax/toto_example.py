"""
Datadog Toto-Open-Base-1.0: Time Series Foundation Model for Observability

Toto é um modelo transformer decoder-only com 151M parâmetros.
Focado em métricas de observabilidade, suporte multivariado nativamente.
"""

import os
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

# Instalar toto-ts se necessário
try:
    from toto.data.util.dataset import MaskedTimeseries
    from toto.inference.forecaster import TotoForecaster
    from toto.model.toto import Toto
    print("Toto já instalado")
except ImportError:
    print("Instalando toto-ts...")
    os.system("pip install toto-ts")
    # Para performance otimizada
    try:
        os.system("pip install xFormers flash-attention")
    except:
        pass
    
    from toto.data.util.dataset import MaskedTimeseries
    from toto.inference.forecaster import TotoForecaster
    from toto.model.toto import Toto

class TotoForecaster:
    def __init__(self, model_name="Datadog/Toto-Open-Base-1.0", device="auto"):
        """
        Inicializa o forecaster Toto
        
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
        self.toto = Toto.from_pretrained(model_name).to(self.device)
        
        # Opcional: compilar para velocidade
        try:
            self.toto.compile()
            print("Modelo compilado para performance otimizada")
        except:
            print("Compilação não disponível, continuando...")
        
        # Inicializar forecaster
        self.forecaster = TotoForecaster(self.toto.model)
        print("Toto forecaster inicializado com sucesso!")
    
    def prepare_multivariate_data(self, data, target_cols, time_col, 
                                series_id_col=None, freq='15min'):
        """
        Prepara dados multivariados para Toto
        
        Args:
            data: DataFrame com os dados
            target_cols: Lista de colunas alvo (métricas)
            time_col: Nome da coluna de tempo
            series_id_col: Nome da coluna de ID da série
            freq: Frequência dos dados (para time_interval_seconds)
        """
        # Ordenar por tempo
        data = data.sort_values(time_col).reset_index(drop=True)
        
        # Converter frequência para segundos
        freq_to_seconds = {
            '1min': 60, '5min': 300, '10min': 600, '15min': 900,
            '30min': 1800, '1h': 3600, '2h': 7200, '1d': 86400
        }
        interval_seconds = freq_to_seconds.get(freq, 900)  # Default 15min
        
        # Extrair séries por ID
        if series_id_col and series_id_col in data.columns:
            unique_ids = data[series_id_col].unique()
            print(f"Encontradas {len(unique_ids)} séries")
            
            series_list = []
            timestamps_list = []
            
            for series_id in unique_ids:
                series_data = data[data[series_id_col] == series_id]
                
                # Extrair valores das métricas
                series_values = series_data[target_cols].values.T  # Transpor para (features, time)
                series_list.append(torch.tensor(series_values, dtype=torch.float32))
                
                # Timestamps
                timestamps = pd.to_datetime(series_data[time_col])
                timestamp_seconds = torch.tensor(
                    timestamps.astype(np.int64) // 10**9, 
                    dtype=torch.float32
                )
                timestamps_list.append(timestamp_seconds)
            
            # Concatenar todas as séries
            input_series = torch.cat(series_list, dim=1).to(self.device)
            timestamp_seconds = torch.cat(timestamps_list, dim=1).to(self.device)
            
        else:
            # Única série
            series_data = data
            input_series = torch.tensor(
                series_data[target_cols].values.T,  # (features, time)
                dtype=torch.float32
            ).to(self.device)
            
            timestamps = pd.to_datetime(series_data[time_col])
            timestamp_seconds = torch.tensor(
                timestamps.astype(np.int64) // 10**9,
                dtype=torch.float32
            ).to(self.device)
        
        # Criar masks
        n_features, n_timesteps = input_series.shape
        padding_mask = torch.full_like(input_series, True, dtype=torch.bool)
        id_mask = torch.zeros_like(input_series)
        
        # Time interval (uniforme para toda a série)
        time_interval_seconds = torch.full((n_features, 1), interval_seconds, 
                                         dtype=torch.float32).to(self.device)
        
        print(f"Dados multivariados preparados:")
        print(f"- Shape: {input_series.shape}")
        print(f"- Features: {target_cols}")
        print(f"- Frequência: {freq} ({interval_seconds}s)")
        print(f"- Período: {len(input_series[0])} timesteps")
        
        return {
            'input_series': input_series,
            'padding_mask': padding_mask,
            'id_mask': id_mask,
            'timestamp_seconds': timestamp_seconds,
            'time_interval_seconds': time_interval_seconds
        }
    
    def create_masked_timeseries(self, data_dict):
        """
        Cria MaskedTimeseries a partir dos dados preparados
        """
        masked_data = MaskedTimeseries(
            series=data_dict['input_series'],
            padding_mask=data_dict['padding_mask'],
            id_mask=data_dict['id_mask'],
            timestamp_seconds=data_dict['timestamp_seconds'],
            time_interval_seconds=data_dict['time_interval_seconds']
        )
        return masked_data
    
    def forecast(self, masked_data, prediction_length=336, num_samples=256, 
                samples_per_batch=256):
        """
        Gera forecast usando Toto
        
        Args:
            masked_data: MaskedTimeseries object
            prediction_length: Passos para prever
            num_samples: Número de samples para forecast probabilístico
            samples_per_batch: Samples por batch
            
        Returns:
            Objeto forecast com previsões
        """
        print(f"Gerando forecast:")
        print(f"- Prediction length: {prediction_length}")
        print(f"- Num samples: {num_samples}")
        print(f"- Samples per batch: {samples_per_batch}")
        
        # Fazer forecast
        forecast = self.forecaster.forecast(
            masked_data,
            prediction_length=prediction_length,
            num_samples=num_samples,
            samples_per_batch=samples_per_batch
        )
        
        print("Forecast gerado com sucesso!")
        print(f"- Median prediction shape: {forecast.median.shape}")
        print(f"- Samples shape: {forecast.samples.shape}")
        
        return forecast

def create_observability_data(n_timesteps=2000, n_services=2, freq='15min'):
    """
    Cria dados sintéticos de observabilidade para demonstração
    """
    np.random.seed(42)
    data_list = []
    
    # Gerar timestamps
    freq_to_minutes = {
        '1min': 1, '5min': 5, '10min': 10, '15min': 15,
        '30min': 30, '1h': 60, '2h': 120, '1d': 1440
    }
    minutes = freq_to_minutes.get(freq, 15)
    
    dates = pd.date_range('2024-01-01', 
                         periods=n_timesteps, 
                         freq=f'{minutes}min')
    
    services = [f'service_{i}' for i in range(n_services)]
    metrics = ['cpu_usage', 'memory_usage', 'request_rate', 'error_rate', 'response_time']
    
    for service in services:
        for i, date in enumerate(dates):
            # Métricas com padrões realistas
            # CPU usage (picos durante horário comercial)
            hour = date.hour
            cpu_base = 20 if 9 <= hour <= 17 else 10
            cpu_pattern = 10 * np.sin(2 * np.pi * hour / 24) if 9 <= hour <= 17 else 0
            cpu_usage = cpu_base + cpu_pattern + np.random.normal(0, 2)
            
            # Memory usage (crescimento gradual com picos)
            memory_base = 30 + i * 0.01  # Leak gradual
            memory_usage = memory_base + 5 * np.sin(2 * np.pi * i / (24 * 4)) + np.random.normal(0, 1)
            
            # Request rate (padrão diário forte)
            request_base = 100 if 9 <= hour <= 17 else 30
            request_rate = request_base + 20 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 5)
            
            # Error rate (baixo, com alguns picos)
            error_base = 0.1
            error_pike = 2.0 if np.random.random() < 0.02 else 0
            error_rate = error_base + error_pike + np.random.exponential(0.05)
            
            # Response time (relacionado a error rate)
            response_time = 50 + error_rate * 20 + np.random.normal(0, 5)
            
            data_list.append({
                'timestamp': date,
                'service': service,
                'cpu_usage': max(0, cpu_usage),
                'memory_usage': max(0, memory_usage),
                'request_rate': max(0, request_rate),
                'error_rate': max(0, error_rate),
                'response_time': max(0, response_time)
            })
    
    return pd.DataFrame(data_list)

def example_single_service_forecasting():
    """
    Exemplo de forecasting para um único serviço
    """
    print("=== Exemplo 1: Forecasting para Serviço Único ===")
    
    # Criar dados de observabilidade
    data = create_observability_data(n_timesteps=1000, n_services=1, freq='15min')
    
    print("Dados de observabilidade criados:")
    print(data.head())
    print(f"Shape: {data.shape}")
    print(f"Serviços: {data['service'].unique()}")
    print(f"Métricas: {[col for col in data.columns if col not in ['timestamp', 'service']]}")
    
    # Inicializar forecaster
    forecaster = TotoForecaster(
        model_name="Datadog/Toto-Open-Base-1.0",
        device="auto"
    )
    
    # Preparar dados
    target_cols = ['cpu_usage', 'memory_usage', 'request_rate', 'error_rate', 'response_time']
    
    data_dict = forecaster.prepare_multivariate_data(
        data,
        target_cols=target_cols,
        time_col='timestamp',
        series_id_col='service',
        freq='15min'
    )
    
    # Criar MaskedTimeseries
    masked_data = forecaster.create_masked_timeseries(data_dict)
    
    # Dividir dados (usar últimos pontos para teste)
    train_size = int(0.9 * len(masked_data.series[0]))
    
    # Forecast para os próximos 48 timesteps (12 horas)
    prediction_length = 48
    
    print(f"\\nConfiguração:")
    print(f"- Contexto: {train_size} timesteps")
    print(f"- Forecast: {prediction_length} timesteps")
    
    # Fazer forecast
    forecast = forecaster.forecast(
        masked_data,
        prediction_length=prediction_length,
        num_samples=128,  # Menos samples para demo
        samples_per_batch=128
    )
    
    # Analisar resultados
    print(f"\\nResultados do forecast:")
    print(f"- Median shape: {forecast.median.shape}")
    print(f"- Samples shape: {forecast.samples.shape}")
    
    # Mostrar previsões para cada métrica
    metric_names = target_cols
    for i, metric in enumerate(metric_names):
        median_pred = forecast.median[i]
        mean_pred = np.mean(forecast.samples[i], axis=0)
        
        print(f"\\n{metric}:")
        print(f"  - Mediana (primeiros 5): {median_pred[:5]}")
        print(f"  - Média (primeiros 5): {mean_pred[:5]}")
        print(f"  - IC 90%: [{forecast.quantile(0.05)[i][0]:.2f}, {forecast.quantile(0.95)[i][0]:.2f}]")
    
    return forecast

def example_multiple_services_forecasting():
    """
    Exemplo de forecasting para múltiplos serviços
    """
    print("\\n=== Exemplo 2: Forecasting para Múltiplos Serviços ===")
    
    # Criar dados com 3 serviços
    data = create_observability_data(n_timesteps=800, n_services=3, freq='10min')
    
    print("Dados multivariados:")
    print(data.head())
    print(f"Shape: {data.shape}")
    print(f"Serviços: {list(data['service'].unique())}")
    
    # Inicializar forecaster
    forecaster = TotoForecaster(
        model_name="Datadog/Toto-Open-Base-1.0"
    )
    
    # Preparar dados
    target_cols = ['cpu_usage', 'memory_usage', 'request_rate', 'error_rate']
    
    data_dict = forecaster.prepare_multivariate_data(
        data,
        target_cols=target_cols,
        time_col='timestamp',
        series_id_col='service',
        freq='10min'
    )
    
    masked_data = forecaster.create_masked_timeseries(data_dict)
    
    # Forecast
    prediction_length = 24
    forecast = forecaster.forecast(
        masked_data,
        prediction_length=prediction_length,
        num_samples=100
    )
    
    print(f"\\nForecast multivariado:")
    print(f"- Serviços: {len(forecast.median[0])}")
    print(f"- Métricas: {len(target_cols)}")
    print(f"- Horizonte: {prediction_length}")
    
    # Analisar por serviço
    n_services = len(data['service'].unique())
    n_timesteps, n_features = forecast.median.shape
    
    # Reorganizar para análise por serviço
    service_results = {}
    for service_idx in range(n_services):
        service_id = data['service'].unique()[service_idx]
        service_predictions = {}
        
        for metric_idx, metric in enumerate(target_cols):
            # Pegar dados para este serviço
            start_idx = service_idx * n_features + metric_idx
            step = n_services * n_features
            
            service_predictions[metric] = {
                'median': forecast.median[start_idx::step][0],  # Primeiro timestep
                'lower_q': forecast.quantile(0.1)[start_idx::step][0],
                'upper_q': forecast.quantile(0.9)[start_idx::step][0]
            }
        
        service_results[service_id] = service_predictions
    
    # Mostrar resultados
    for service, results in service_results.items():
        print(f"\\n{service}:")
        for metric, preds in results.items():
            print(f"  {metric}: {preds['median']:.2f} [{preds['lower_q']:.2f}, {preds['upper_q']:.2f}]")
    
    return service_results

def example_observability_with_external_factors():
    """
    Exemplo com fatores externos (simulando covariáveis)
    """
    print("\\n=== Exemplo 3: Observabilidade com Fatores Externos ===")
    print("Simulando como covariáveis afetariam métricas de observabilidade")
    
    # Criar dados base
    base_data = create_observability_data(n_timesteps=600, n_services=1, freq='1h')
    
    # Adicionar fatores externos
    np.random.seed(42)
    external_factors = pd.DataFrame({
        'timestamp': base_data['timestamp'],
        'traffic_spike': np.random.binomial(1, 0.1, len(base_data)),
        'deployment': np.random.binomial(1, 0.05, len(base_data)),
        'day_of_week': base_data['timestamp'].dt.dayofweek,
        'hour': base_data['timestamp'].dt.hour
    })
    
    print("Fatores externos adicionados:")
    print(external_factors.head())
    print(f"\\nCorrelação com métricas:")
    
    # Calcular correlações
    target_cols = ['cpu_usage', 'memory_usage', 'request_rate']
    
    for col in ['traffic_spike', 'deployment', 'day_of_week', 'hour']:
        for metric in target_cols:
            correlation = base_data[metric].corr(external_factors[col])
            print(f"- {metric} vs {col}: {correlation:.3f}")
    
    # Inicializar forecaster
    forecaster = TotoForecaster(
        model_name="Datadog/Toto-Open-Base-1.0"
    )
    
    # Preparar dados (sem fatores externos - Toto é multivariado)
    data_dict = forecaster.prepare_multivariate_data(
        base_data,
        target_cols=target_cols,
        time_col='timestamp',
        series_id_col='service',
        freq='1h'
    )
    
    masked_data = forecaster.create_masked_timeseries(data_dict)
    
    # Forecast
    prediction_length = 72  # 3 dias
    forecast = forecaster.forecast(
        masked_data,
        prediction_length=prediction_length,
        num_samples=150
    )
    
    print(f"\\nForecast com fatores externos simulados:")
    print(f"- Horizonte: {prediction_length} horas")
    
    # Análise de tendências
    for i, metric in enumerate(target_cols):
        trend = np.diff(forecast.median[i])
        upward_trend = np.sum(trend > 0) / len(trend)
        
        print(f"\\n{metric}:")
        print(f"  - Tendência ascendente: {upward_trend:.1%}")
        print(f"  - Média forecast: {np.mean(forecast.median[i]):.2f}")
        print(f"  - IC 95%: [{np.mean(forecast.quantile(0.025)[i]):.2f}, {np.mean(forecast.quantile(0.975)[i]):.2f}]")
    
    return forecast, external_factors

if __name__ == "__main__":
    print("=== Datadog Toto Time Series Forecasting ===\\n")
    
    try:
        # Exemplo 1: Serviço único
        forecast1 = example_single_service_forecasting()
        
        # Exemplo 2: Múltiplos serviços
        results2 = example_multiple_services_forecasting()
        
        # Exemplo 3: Fatores externos
        forecast3, factors3 = example_observability_with_external_factors()
        
        print("\\n=== Resumo ===")
        print("✓ Exemplo 1: Forecasting para serviço único")
        print("✓ Exemplo 2: Forecasting para múltiplos serviços")
        print("✓ Exemplo 3: Observabilidade com fatores externos")
        print("\\nToto é ideal para:")
        print("- Métricas de observabilidade")
        print("- Sistemas multivariados")
        print("- Forecasts probabilísticos")
        print("- Dados de alta dimensionalidade")
        
    except Exception as e:
        print(f"Erro durante execução: {e}")
        print("\\nPara usar Toto, certifique-se de que está instalado:")
        print("pip install toto-ts")
        print("Opcional: pip install xFormers flash-attention")