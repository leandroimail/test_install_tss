"""
IBM PatchTST: Transformer-based Time Series Foundation Model

PatchTST é um modelo transformer que trata cada canal como série univariada independente.
Implementação com suporte para forecasting, classificação e regressão.
"""

import os
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

# Instalar transformers se necessário
try:
    from transformers import PatchTSTConfig, PatchTSTForPrediction
    print("Transformers já instalado")
except ImportError:
    print("Instalando transformers...")
    os.system("pip install transformers")
    from transformers import PatchTSTConfig, PatchTSTForPrediction

class PatchTSTForecaster:
    def __init__(self, model_name="ibm-granite/granite-timeseries-patchtst", 
                 device="auto", context_length=512, prediction_length=96):
        """
        Inicializa o forecaster PatchTST
        
        Args:
            model_name: Nome do modelo no Hugging Face
            device: Dispositivo de execução
            context_length: Comprimento do contexto
            prediction_length: Comprimento da previsão
        """
        self.model_name = model_name
        self.device = device
        self.context_length = context_length
        self.prediction_length = prediction_length
        
        # Configurar device
        if device == "auto":
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Carregando modelo PatchTST {model_name} no device {self.device}...")
        
        # Configurar modelo
        config = PatchTSTConfig(
            context_length=context_length,
            prediction_length=prediction_length,
            output_attentions=False,
            dropout=0.1,
            num_input_channels=1,  # Para univariado, será ajustado dinamicamente
        )
        
        # Carregar modelo
        self.model = PatchTSTForPrediction.from_pretrained(
            model_name,
            config=config
        ).to(self.device)
        
        self.model.eval()
        print("PatchTST model carregado com sucesso!")
    
    def prepare_univariate_data(self, data, target_col, time_col, series_id_col=None):
        """
        Prepara dados univariados para PatchTST
        
        Args:
            data: DataFrame com os dados
            target_col: Nome da coluna alvo
            time_col: Nome da coluna de tempo
            series_id_col: Nome da coluna de ID da série
        """
        # Ordenar por tempo
        data = data.sort_values(time_col).reset_index(drop=True)
        
        # Extrair série
        if series_id_col and series_id_col in data.columns:
            unique_ids = data[series_id_col].unique()
            series_data = data[data[series_id_col] == unique_ids[0]]
            print(f"Usando série: {unique_ids[0]}")
        else:
            series_data = data
        
        # Extrair valores
        time_series = series_data[target_col].values
        timestamps = pd.to_datetime(series_data[time_col])
        
        print(f"Dados univariados preparados:")
        print(f"- Série shape: {time_series.shape}")
        print(f"- Período: {timestamps.min()} a {timestamps.max()}")
        
        return time_series, timestamps
    
    def prepare_multivariate_data(self, data, target_cols, time_col, series_id_col=None):
        """
        Prepara dados multivariados para PatchTST
        Nota: PatchTST trata cada canal independentemente, então procesamos um por vez
        
        Args:
            data: DataFrame com os dados
            target_cols: Lista de colunas alvo
            time_col: Nome da coluna de tempo
            series_id_col: Nome da coluna de ID da série
        """
        # Ordenar por tempo
        data = data.sort_values(time_col).reset_index(drop=True)
        
        # Extrair séries
        if series_id_col and series_id_col in data.columns:
            unique_ids = data[series_id_col].unique()
            series_data = data[data[series_id_col] == unique_ids[0]]
            print(f"Usando série: {unique_ids[0]}")
        else:
            series_data = data
        
        # Extrair cada coluna alvo
        series_dict = {}
        for col in target_cols:
            if col in series_data.columns:
                series_dict[col] = series_data[col].values
        
        print(f"Dados multivariados preparados (tratarão independentemente):")
        print(f"- Colunas: {list(series_dict.keys())}")
        for col, series in series_dict.items():
            print(f"  - {col}: {series.shape}")
        
        return series_dict
    
    def create_patches(self, time_series, patch_length=16, patch_stride=8):
        """
        Cria patches para PatchTST
        
        Args:
            time_series: Série temporal
            patch_length: Tamanho do patch
            patch_stride: Stride entre patches
            
        Returns:
            Patches criados
        """
        n_samples = len(time_series)
        n_patches = (n_samples - patch_length) // patch_stride + 1
        
        patches = []
        for i in range(n_patches):
            start_idx = i * patch_stride
            end_idx = start_idx + patch_length
            if end_idx <= n_samples:
                patch = time_series[start_idx:end_idx]
                patches.append(patch)
        
        print(f"Patches criados:")
        print(f"- Num patches: {len(patches)}")
        print(f"- Patch length: {patch_length}")
        print(f"- Patch stride: {patch_stride}")
        
        return np.array(patches)
    
    def forecast_univariate(self, time_series, patch_length=16, patch_stride=8):
        """
        Forecast univariado usando PatchTST
        
        Args:
            time_series: Série temporal
            patch_length: Tamanho do patch
            patch_stride: Stride entre patches
            
        Returns:
            Previsões
        """
        print(f"Gerando forecast univariado:")
        print(f"- Série shape: {time_series.shape}")
        
        # Ajustar configuração se necessário
        if self.model.config.context_length != len(time_series) - self.prediction_length:
            print(f"Ajustando context_length para {len(time_series) - self.prediction_length}")
            self.model.config.context_length = len(time_series) - self.prediction_length
        
        # Dividir série em contexto e target
        context_length = len(time_series) - self.prediction_length
        context_values = time_series[:context_length]
        target_values = time_series[context_length:]
        
        # Criar tensor de entrada
        input_tensor = torch.tensor(
            context_values.reshape(1, -1, 1),  # (batch, context, channels)
            dtype=torch.float32,
            device=self.device
        )
        
        # Criar past time features (simplificado)
        past_time_features = torch.zeros(
            (1, context_length, self.model.config.num_time_features),
            dtype=torch.float32,
            device=self.device
        )
        
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Fazer forecast
        with torch.no_grad():
            outputs = self.model(
                past_values=input_tensor,
                past_time_features=past_time_features
            )
        
        # Extrair previsões
        predictions = outputs.prediction_outputs.logits.squeeze(0).cpu().numpy()
        
        print(f"Forecast shape: {predictions.shape}")
        print("Forecast univariado gerado com sucesso!")
        
        return predictions, target_values
    
    def forecast_multivariate(self, series_dict, patch_length=16, patch_stride=8):
        """
        Forecast multivariado (cada canal independente)
        
        Args:
            series_dict: Dicionário com séries por canal
            patch_length: Tamanho do patch
            patch_stride: Stride entre patches
            
        Returns:
            Previsões para cada canal
        """
        print(f"Gerando forecast multivariado (canais independentes):")
        print(f"- Canais: {list(series_dict.keys())}")
        
        all_results = {}
        
        for channel, time_series in series_dict.items():
            print(f"\\nProcessando canal: {channel}")
            
            # Ajustar configuração para este canal
            self.model.config.num_input_channels = 1
            self.model.config.context_length = len(time_series) - self.prediction_length
            
            # Fazer forecast
            predictions, target = self.forecast_univariate(time_series, patch_length, patch_stride)
            
            all_results[channel] = {
                'predictions': predictions,
                'target': target,
                'mae': np.mean(np.abs(predictions - target)),
                'rmse': np.sqrt(np.mean((predictions - target)**2))
            }
            
            print(f"  - MAE: {all_results[channel]['mae']:.2f}")
            print(f"  - RMSE: {all_results[channel]['rmse']:.2f}")
        
        return all_results

def create_patchtst_data(n_timesteps=800, n_channels=3, n_series=1):
    """
    Cria dados sintéticos para PatchTST
    """
    np.random.seed(42)
    data_list = []
    
    for series_id in range(n_series):
        # Gerar timestamps
        dates = pd.date_range('2020-01-01', periods=n_timesteps, freq='D')
        
        # Gerar múltiplos canais correlacionados
        # Canal 1: Temperatura
        temp_base = 20 + 10 * np.sin(2 * np.pi * np.arange(n_timesteps) / 365.25)
        temp_seasonal = 5 * np.sin(2 * np.pi * np.arange(n_timesteps) / 7)  # Weekly
        temperature = temp_base + temp_seasonal + np.random.normal(0, 2, n_timesteps)
        
        # Canal 2: Humidade (correlacionada com temperatura)
        humidity_base = 60 - 0.5 * (temp_base - 20)  # Correlação negativa
        humidity = humidity_base + np.random.normal(0, 5, n_timesteps)
        
        # Canal 3: Pressão (independente)
        pressure_base = 1013 + 10 * np.sin(2 * np.pi * np.arange(n_timesteps) / 30)  # Monthly
        pressure = pressure_base + np.random.normal(0, 3, n_timesteps)
        
        # Canal 4: Velocidade do vento (independente)
        wind_base = 5 + 2 * np.sin(2 * np.pi * np.arange(n_timesteps) / 24)  # Daily
        wind_speed = np.maximum(0, wind_base + np.random.exponential(2, n_timesteps))
        
        for i, date in enumerate(dates):
            data_list.append({
                'timestamp': date,
                'temperature': temperature[i],
                'humidity': humidity[i],
                'pressure': pressure[i],
                'wind_speed': wind_speed[i],
                'series_id': f'station_{series_id}'
            })
    
    return pd.DataFrame(data_list)

def example_univariate_forecasting():
    """
    Exemplo de forecasting univariado com PatchTST
    """
    print("=== Exemplo 1: Forecasting Univariado ===")
    
    # Criar dados sintéticos
    data = create_patchtst_data(n_timesteps=600, n_channels=1, n_series=1)
    
    print("Dados univariados criados:")
    print(data.head())
    print(f"Shape: {data.shape}")
    
    # Inicializar forecaster
    forecaster = PatchTSTForecaster(
        model_name="ibm-granite/granite-timeseries-patchtst",
        context_length=512,
        prediction_length=96
    )
    
    # Preparar dados
    time_series, timestamps = forecaster.prepare_univariate_data(
        data,
        target_col='temperature',
        time_col='timestamp',
        series_id_col='series_id'
    )
    
    # Dividir dados
    train_size = int(0.85 * len(time_series))
    train_series = time_series[:train_size]
    full_series = time_series  # Para forecast completo
    
    print(f"\\nConfiguração:")
    print(f"- Série completa: {len(full_series)} pontos")
    print(f"- Contexto: {len(train_series)} pontos")
    print(f"- Predição: {forecaster.prediction_length} pontos")
    
    # Forecast
    predictions, actual = forecaster.forecast_univariate(full_series)
    
    # Avaliar
    mae = np.mean(np.abs(predictions - actual))
    rmse = np.sqrt(np.mean((predictions - actual)**2))
    
    print(f"\\nResultados:")
    print(f"- MAE: {mae:.2f}")
    print(f"- RMSE: {rmse:.2f}")
    print(f"- Previsões (primeiras 5): {predictions[:5]}")
    print(f"- Reais (primeiras 5): {actual[:5]}")
    
    return predictions, actual

def example_multivariate_forecasting():
    """
    Exemplo de forecasting multivariado
    """
    print("\\n=== Exemplo 2: Forecasting Multivariado ===")
    
    # Criar dados com múltiplos canais
    data = create_patchtst_data(n_timesteps=500, n_channels=4, n_series=1)
    
    print("Dados multivariados:")
    print(data.head())
    print(f"Shape: {data.shape}")
    print(f"Colunas: {list(data.columns)}")
    
    # Inicializar forecaster
    forecaster = PatchTSTForecaster(
        model_name="ibm-granite/granite-timeseries-patchtst",
        context_length=256,
        prediction_length=48
    )
    
    # Preparar dados multivariados (PatchTST trata cada canal independentemente)
    series_dict = forecaster.prepare_multivariate_data(
        data,
        target_cols=['temperature', 'humidity', 'pressure', 'wind_speed'],
        time_col='timestamp',
        series_id_col='series_id'
    )
    
    # Fazer forecast para cada canal
    all_results = forecaster.forecast_multivariate(series_dict)
    
    print(f"\\nResumo dos resultados:")
    for channel, results in all_results.items():
        print(f"- {channel}: MAE={results['mae']:.2f}, RMSE={results['rmse']:.2f}")
    
    return all_results

def example_patch_analysis():
    """
    Exemplo de análise de patches
    """
    print("\\n=== Exemplo 3: Análise de Patches ===")
    
    # Criar dados com padrões claros
    np.random.seed(42)
    n_timesteps = 400
    dates = pd.date_range('2020-01-01', periods=n_timesteps, freq='D')
    
    # Série com padrões sazonais múltiplos
    daily_pattern = 10 * np.sin(2 * np.pi * np.arange(n_timesteps) / 1)  # Daily
    weekly_pattern = 15 * np.sin(2 * np.pi * np.arange(n_timesteps) / 7)  # Weekly
    monthly_pattern = 20 * np.sin(2 * np.pi * np.arange(n_timesteps) / 30)  # Monthly
    trend = np.linspace(100, 120, n_timesteps)
    noise = np.random.normal(0, 2, n_timesteps)
    
    time_series = daily_pattern + weekly_pattern + monthly_pattern + trend + noise
    
    data = pd.DataFrame({
        'timestamp': dates,
        'complex_series': time_series
    })
    
    print("Série complexa criada:")
    print(f"- Padrões: daily, weekly, monthly, trend")
    print(f"- Shape: {data.shape}")
    
    # Inicializar forecaster
    forecaster = PatchTSTForecaster(
        model_name="ibm-granite/granite-timeseries-patchtst",
        context_length=200,
        prediction_length=40
    )
    
    # Extrair série
    time_series, _ = forecaster.prepare_univariate_data(
        data,
        target_col='complex_series',
        time_col='timestamp'
    )
    
    # Testar diferentes configurações de patch
    patch_configs = [
        {'length': 8, 'stride': 4},
        {'length': 16, 'stride': 8},
        {'length': 32, 'stride': 16},
        {'length': 64, 'stride': 32}
    ]
    
    results = {}
    
    for config in patch_configs:
        print(f"\\nTestando patches: length={config['length']}, stride={config['stride']}")
        
        # Criar patches
        patches = forecaster.create_patches(
            time_series,
            patch_length=config['length'],
            patch_stride=config['stride']
        )
        
        print(f"  - Patches criados: {len(patches)}")
        
        # Fazer forecast
        predictions, actual = forecaster.forecast_univariate(
            time_series,
            patch_length=config['length'],
            patch_stride=config['stride']
        )
        
        # Métricas
        mae = np.mean(np.abs(predictions - actual))
        rmse = np.sqrt(np.mean((predictions - actual)**2))
        
        results[f"patch_{config['length']}_{config['stride']}"] = {
            'predictions': predictions,
            'actual': actual,
            'mae': mae,
            'rmse': rmse,
            'n_patches': len(patches)
        }
        
        print(f"  - MAE: {mae:.2f}")
        print(f"  - RMSE: {rmse:.2f}")
    
    # Encontrar melhor configuração
    best_config = min(results.keys(), key=lambda x: results[x]['mae'])
    print(f"\\nMelhor configuração: {best_config}")
    print(f"  - MAE: {results[best_config]['mae']:.2f}")
    print(f"  - RMSE: {results[best_config]['rmse']:.2f}")
    
    return results

def example_channel_independence():
    """
    Exemplo demonstrando independência dos canais
    """
    print("\\n=== Exemplo 4: Independência dos Canais ===")
    
    # Criar dados com canais claramente independentes
    np.random.seed(42)
    n_timesteps = 300
    
    # Canal 1: Padrão sinusoidal
    channel1 = 100 + 20 * np.sin(2 * np.pi * np.arange(n_timesteps) / 50)
    
    # Canal 2: Ruído com deriva
    channel2 = 50 + 0.1 * np.arange(n_timesteps) + np.random.normal(0, 5, n_timesteps)
    
    # Canal 3: Padrão quadrático
    channel3 = 200 - 0.001 * (np.arange(n_timesteps) - 150)**2 + np.random.normal(0, 3, n_timesteps)
    
    # Canal 4: Cíclico curto
    channel4 = 75 + 10 * np.sin(2 * np.pi * np.arange(n_timesteps) / 10)
    
    data = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=n_timesteps, freq='D'),
        'sinusoidal': channel1,
        'drift_noise': channel2,
        'quadratic': channel3,
        'short_cycle': channel4
    })
    
    print("Dados com canais independentes:")
    print(data.head())
    print(f"Shape: {data.shape}")
    
    # Inicializar forecaster
    forecaster = PatchTSTForecaster(
        model_name="ibm-granite/granite-timeseries-patchtst",
        context_length=150,
        prediction_length=30
    )
    
    # Preparar dados
    series_dict = forecaster.prepare_multivariate_data(
        data,
        target_cols=['sinusoidal', 'drift_noise', 'quadratic', 'short_cycle'],
        time_col='timestamp'
    )
    
    # Fazer forecast
    results = forecaster.forecast_multivariate(series_dict)
    
    # Análise de independência
    print(f"\\nAnálise de independência dos canais:")
    correlation_matrix = data[['sinusoidal', 'drift_noise', 'quadratic', 'short_cycle']].corr()
    print("Correlações originais:")
    print(correlation_matrix.round(3))
    
    # Análise das previsões
    pred_correlations = np.corrcoef([
        results['sinusoidal']['predictions'],
        results['drift_noise']['predictions'],
        results['quadratic']['predictions'],
        results['short_cycle']['predictions']
    ])
    
    print("\\nCorrelações nas previsões:")
    pred_corr_df = pd.DataFrame(
        pred_correlations,
        index=['sinusoidal', 'drift_noise', 'quadratic', 'short_cycle'],
        columns=['sinusoidal', 'drift_noise', 'quadratic', 'short_cycle']
    )
    print(pred_corr_df.round(3))
    
    # Verificar se independência foi preservada
    print(f"\\nVerificação da independência:")
    independent_pairs = [('sinusoidal', 'drift_noise'), ('quadratic', 'short_cycle')]
    for pair in independent_pairs:
        orig_corr = abs(correlation_matrix.loc[pair[0], pair[1]])
        pred_corr = abs(pred_corr_df.loc[pair[0], pair[1]])
        print(f"- {pair[0]} vs {pair[1]}: Original={orig_corr:.3f}, Previsto={pred_corr:.3f}")
    
    return results

if __name__ == "__main__":
    print("=== IBM PatchTST Time Series Forecasting ===\\n")
    
    try:
        # Exemplo 1: Univariado
        pred1, actual1 = example_univariate_forecasting()
        
        # Exemplo 2: Multivariado
        results2 = example_multivariate_forecasting()
        
        # Exemplo 3: Análise de patches
        results3 = example_patch_analysis()
        
        # Exemplo 4: Independência dos canais
        results4 = example_channel_independence()
        
        print("\\n=== Resumo ===")
        print("✓ Exemplo 1: Forecasting univariado")
        print("✓ Exemplo 2: Forecasting multivariado (canais independentes)")
        print("✓ Exemplo 3: Análise de diferentes configurações de patch")
        print("✓ Exemplo 4: Demonstração da independência dos canais")
        print("\\nPatchTST é ideal para:")
        print("- Canais independentes (séries univariadas)")
        print("- Long-term forecasting")
        print("- Transfer learning entre canais")
        print("- Eficiência com patching (redução de complexidade)")
        print("- Arquitetura transformer otimizada para séries temporais")
        
    except Exception as e:
        print(f"Erro durante execução: {e}")
        print("\\nPara usar PatchTST, certifique-se de que transformers está instalado:")
        print("pip install transformers")
        print("\\nE que o modelo ibm-granite/granite-timeseries-patchtst está disponível no Hugging Face")