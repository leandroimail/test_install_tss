"""
IBM PatchTSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting

PatchTSMixer é um modelo multivariado baseado em MLP-Mixer para forecasting,
com suporte para mistura entre patches, canais e features ocultas.
"""

import os
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

# Instalar transformers se necessário
try:
    from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction
    print("Transformers já instalado")
except ImportError:
    print("Instalando transformers...")
    os.system("pip install transformers")
    from transformers import PatchTSMixerConfig, PatchTSMixerForPrediction

class PatchTSMixerForecaster:
    def __init__(self, model_name="ibm-granite/granite-timeseries-patchtsmixer", 
                 device="auto", context_length=512, prediction_length=96):
        """
        Inicializa o forecaster PatchTSMixer
        
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
        
        print(f"Carregando modelo PatchTSMixer {model_name} no device {self.device}...")
        
        # Configurar modelo
        config = PatchTSMixerConfig(
            context_length=context_length,
            prediction_length=prediction_length,
            patch_length=16,
            patch_stride=8,
            num_input_channels=7,  # Ajustado para ETTh1 default
            encoder_dim=256,
            decoder_dim=256,
            encoder_n_heads=8,
            decoder_n_heads=8,
            dropout=0.1,
        )
        
        # Carregar modelo
        self.model = PatchTSMixerForPrediction.from_pretrained(
            model_name,
            config=config
        ).to(self.device)
        
        self.model.eval()
        print("PatchTSMixer model carregado com sucesso!")
    
    def prepare_multivariate_data(self, data, target_cols, time_col, series_id_col=None):
        """
        Prepara dados multivariados para PatchTSMixer
        
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
        
        # Verificar se todas as colunas existem
        available_cols = []
        for col in target_cols:
            if col in series_data.columns:
                available_cols.append(col)
            else:
                print(f"Coluna {col} não encontrada, ignorando")
        
        if not available_cols:
            raise ValueError("Nenhuma coluna alvo encontrada")
        
        # Extrair séries multivariadas
        multivariate_data = series_data[available_cols].values  # (time, channels)
        
        # Ajustar configuração do modelo
        self.model.config.num_input_channels = len(available_cols)
        
        print(f"Dados multivariados preparados:")
        print(f"- Shape: {multivariate_data.shape}")
        print(f"- Colunas: {available_cols}")
        print(f"- Channels: {self.model.config.num_input_channels}")
        
        return multivariate_data, available_cols
    
    def create_patches(self, multivariate_data, patch_length=16, patch_stride=8):
        """
        Cria patches para PatchTSMixer
        
        Args:
            multivariate_data: Dados multivariados (time, channels)
            patch_length: Tamanho do patch
            patch_stride: Stride entre patches
            
        Returns:
            Patches criados
        """
        n_samples, n_channels = multivariate_data.shape
        n_patches = (n_samples - patch_length) // patch_stride + 1
        
        patches = []
        for i in range(n_patches):
            start_idx = i * patch_stride
            end_idx = start_idx + patch_length
            if end_idx <= n_samples:
                patch = multivariate_data[start_idx:end_idx, :]  # (patch_length, channels)
                patches.append(patch)
        
        patches = np.array(patches)  # (n_patches, patch_length, channels)
        
        print(f"Patches criados:")
        print(f"- Num patches: {len(patches)}")
        print(f"- Patch shape: {patches.shape}")
        
        return patches
    
    def forecast(self, multivariate_data, patch_length=16, patch_stride=8):
        """
        Gera forecast usando PatchTSMixer
        
        Args:
            multivariate_data: Dados multivariados (time, channels)
            patch_length: Tamanho do patch
            patch_stride: Stride entre patches
            
        Returns:
            Previsões para cada canal
        """
        print(f"Gerando forecast multivariado:")
        print(f"- Data shape: {multivariate_data.shape}")
        print(f"- Prediction length: {self.prediction_length}")
        
        # Ajustar configuração se necessário
        context_length = len(multivariate_data) - self.prediction_length
        self.model.config.context_length = context_length
        self.model.config.patch_length = patch_length
        self.model.config.patch_stride = patch_stride
        
        # Dividir dados
        context_data = multivariate_data[:context_length]  # (context_length, channels)
        target_data = multivariate_data[context_length:]   # (prediction_length, channels)
        
        # Criar tensor de entrada
        input_tensor = torch.tensor(
            context_data.reshape(1, context_length, -1),  # (1, context, channels)
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
        print("Forecast multivariado gerado com sucesso!")
        
        return predictions, target_data

def create_patchtsmixer_data(n_timesteps=600, n_channels=5, n_series=1):
    """
    Cria dados sintéticos para PatchTSMixer
    """
    np.random.seed(42)
    data_list = []
    
    for series_id in range(n_series):
        # Gerar timestamps
        dates = pd.date_range('2020-01-01', periods=n_timesteps, freq='H')
        
        # Simular dados de energia (similar ao ETTh1)
        # Canal 1: High Usage (HUFL)
        hour_pattern = 100 + 50 * np.sin(2 * np.pi * np.arange(n_timesteps) / 24)  # Daily
        daily_trend = 20 * np.sin(2 * np.pi * np.arange(n_timesteps) / (24 * 7))  # Weekly
        noise = np.random.normal(0, 5, n_timesteps)
        hufl = hour_pattern + daily_trend + noise
        
        # Canal 2: High Usage (HULL) - correlacionado com HUFL
        hulf = 0.8 * hufl + 0.2 * np.random.normal(100, 20, n_timesteps)
        
        # Canal 3: Medium Usage (MUFL) - correlacionado mas menos intenso
        mufl = 0.6 * hufl + 0.4 * np.random.normal(60, 15, n_timesteps)
        
        # Canal 4: Medium Usage (MULL) - relacionado a MUFL
        mull = 0.9 * mufl + 0.1 * np.random.normal(50, 10, n_timesteps)
        
        # Canal 5: Low Usage (LUFL) - padrão diferente
        lufl = 40 + 20 * np.sin(2 * np.pi * np.arange(n_timesteps) / 12) + np.random.normal(0, 3, n_timesteps)
        
        # Canal 6: Low Usage (LULL) - correlacionado com LUFL
        lull = 0.7 * lufl + 0.3 * np.random.normal(30, 8, n_timesteps)
        
        # Canal 7: Oil Temperature (OT) - independente mas sazonal
        ot_base = 30 + 10 * np.sin(2 * np.pi * np.arange(n_timesteps) / 365)  # Yearly
        ot_daily = 5 * np.sin(2 * np.pi * np.arange(n_timesteps) / 24)  # Daily
        oil_temp = ot_base + ot_daily + np.random.normal(0, 2, n_timesteps)
        
        # Covariáveis dinâmicas adicionais (se necessário)
        # Temperatura externa
        ext_temp = 20 + 15 * np.sin(2 * np.pi * np.arange(n_timesteps) / 365) + np.random.normal(0, 3, n_timesteps)
        
        # Preço da energia
        price = 0.1 + 0.02 * np.sin(2 * np.pi * np.arange(n_timesteps) / (24 * 7)) + np.random.normal(0, 0.01, n_timesteps)
        
        for i, date in enumerate(dates):
            data_list.append({
                'timestamp': date,
                'HUFL': hufl[i],
                'HULL': hulf[i],
                'MUFL': mufl[i],
                'MULL': mull[i],
                'LUFL': lufl[i],
                'LULL': lull[i],
                'OT': oil_temp[i],
                'ext_temp': ext_temp[i],
                'price': price[i],
                'series_id': f'transformer_{series_id}'
            })
    
    return pd.DataFrame(data_list)

def example_basic_multivariate_forecasting():
    """
    Exemplo básico de forecasting multivariado
    """
    print("=== Exemplo 1: Forecasting Multivariado Básico ===")
    
    # Criar dados sintéticos
    data = create_patchtsmixer_data(n_timesteps=500, n_channels=7, n_series=1)
    
    print("Dados multivariados criados:")
    print(data.head())
    print(f"Shape: {data.shape}")
    print(f"Colunas: {list(data.columns)}")
    
    # Inicializar forecaster
    forecaster = PatchTSMixerForecaster(
        model_name="ibm-granite/granite-timeseries-patchtsmixer",
        context_length=400,
        prediction_length=96
    )
    
    # Preparar dados multivariados
    target_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    
    multivariate_data, used_cols = forecaster.prepare_multivariate_data(
        data,
        target_cols=target_cols,
        time_col='timestamp',
        series_id_col='series_id'
    )
    
    # Forecast
    predictions, actual = forecaster.forecast(multivariate_data)
    
    # Análise dos resultados
    print(f"\\nResultados do forecast:")
    print(f"- Predictions shape: {predictions.shape}")
    print(f"- Actual shape: {actual.shape}")
    
    # Métricas por canal
    channel_results = {}
    for i, channel in enumerate(used_cols):
        pred = predictions[:, i]
        target = actual[:, i]
        
        mae = np.mean(np.abs(pred - target))
        rmse = np.sqrt(np.mean((pred - target)**2))
        
        channel_results[channel] = {
            'predictions': pred,
            'actual': target,
            'mae': mae,
            'rmse': rmse
        }
        
        print(f"\\n{channel}:")
        print(f"  - MAE: {mae:.2f}")
        print(f"  - RMSE: {rmse:.2f}")
        print(f"  - Pred mean: {np.mean(pred):.2f}")
        print(f"  - Actual mean: {np.mean(target):.2f}")
    
    return channel_results

def example_covariate_integration():
    """
    Exemplo integrando covariáveis
    """
    print("\\n=== Exemplo 2: Integração de Covariáveis ===")
    
    # Criar dados com covariáveis
    np.random.seed(42)
    n_timesteps = 400
    dates = pd.date_range('2020-01-01', periods=n_timesteps, freq='H')
    
    # Séries multivariadas base
    base_series = []
    for i in range(5):  # 5 canais
        trend = 100 + i * 20
        seasonal = 30 * np.sin(2 * np.pi * np.arange(n_timesteps) / (24 * 7))  # Weekly
        channel = trend + seasonal + np.random.normal(0, 5, n_timesteps)
        base_series.append(channel)
    
    base_data = np.column_stack(base_series)  # (time, channels)
    
    # Covariáveis dinâmicas
    temperature = 20 + 10 * np.sin(2 * np.pi * np.arange(n_timesteps) / 365) + np.random.normal(0, 2, n_timesteps)
    price = 0.1 + 0.05 * np.sin(2 * np.pi * np.arange(n_timesteps) / 30) + np.random.normal(0, 0.01, n_timesteps)
    demand_index = 1.0 + 0.1 * np.sin(2 * np.pi * np.arange(n_timesteps) / 7) + np.random.normal(0, 0.05, n_timesteps)
    
    # Simular efeito das covariáveis
    # Canal 1: sensível à temperatura
    base_data[:, 0] += 0.5 * temperature
    
    # Canal 2: sensível ao preço
    base_data[:, 1] -= 10 * (price - 0.1)
    
    # Canal 3: sensível ao índice de demanda
    base_data[:, 2] += 5 * (demand_index - 1.0)
    
    # Criar DataFrame
    channel_names = [f'channel_{i+1}' for i in range(5)]
    data = pd.DataFrame({
        'timestamp': dates,
        **{channel_names[i]: base_data[:, i] for i in range(5)},
        'temperature': temperature,
        'price': price,
        'demand_index': demand_index
    })
    
    print("Dados com covariáveis:")
    print(data.head())
    print(f"Shape: {data.shape}")
    
    # Análise de correlações
    print(f"\\nCorrelações com covariáveis:")
    covariate_cols = ['temperature', 'price', 'demand_index']
    for col in covariate_cols:
        print(f"\\n{col}:")
        for channel in channel_names:
            corr = data[channel].corr(data[col])
            print(f"  - {channel}: {corr:.3f}")
    
    # Inicializar forecaster
    forecaster = PatchTSMixerForecaster(
        model_name="ibm-granite/granite-timeseries-patchtsmixer",
        context_length=300,
        prediction_length=48
    )
    
    # Preparar dados (sem covariáveis separadas - PatchTSMixer usa mistura de canais)
    multivariate_data, used_cols = forecaster.prepare_multivariate_data(
        data,
        target_cols=channel_names,
        time_col='timestamp',
        series_id_col=None
    )
    
    # Forecast
    predictions, actual = forecaster.forecast(multivariate_data)
    
    # Análise dos resultados
    print(f"\\nResultados com covariáveis simuladas:")
    print(f"- Predictions shape: {predictions.shape}")
    
    # Métricas
    for i, channel in enumerate(used_cols):
        pred = predictions[:, i]
        target = actual[:, i]
        
        mae = np.mean(np.abs(pred - target))
        rmse = np.sqrt(np.mean((pred - target)**2))
        
        print(f"\\n{channel}:")
        print(f"  - MAE: {mae:.2f}")
        print(f"  - RMSE: {rmse:.2f}")
    
    return predictions, actual

def example_channel_correlation_analysis():
    """
    Exemplo de análise de correlação entre canais
    """
    print("\\n=== Exemplo 3: Análise de Correlação entre Canais ===")
    
    # Criar dados com correlações conhecidas
    np.random.seed(42)
    n_timesteps = 300
    
    # Canal base
    base = 100 + 30 * np.sin(2 * np.pi * np.arange(n_timesteps) / 50) + np.random.normal(0, 3, n_timesteps)
    
    # Canal 1: Altamente correlacionado com base
    channel1 = 0.9 * base + 0.1 * np.random.normal(0, 5, n_timesteps)
    
    # Canal 2: Moderadamente correlacionado
    channel2 = 0.6 * base + 0.4 * np.random.normal(50, 15, n_timesteps)
    
    # Canal 3: Fracamente correlacionado
    channel3 = 0.3 * base + 0.7 * np.random.normal(75, 20, n_timesteps)
    
    # Canal 4: Independente
    channel4 = 80 + 20 * np.sin(2 * np.pi * np.arange(n_timesteps) / 25) + np.random.normal(0, 8, n_timesteps)
    
    data = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=n_timesteps, freq='D'),
        'base': base,
        'high_corr': channel1,
        'med_corr': channel2,
        'low_corr': channel3,
        'independent': channel4
    })
    
    print("Dados com correlações controladas:")
    print(data.head())
    
    # Análise de correlações originais
    correlation_matrix = data[['base', 'high_corr', 'med_corr', 'low_corr', 'independent']].corr()
    print("\\nCorrelações originais:")
    print(correlation_matrix.round(3))
    
    # Inicializar forecaster
    forecaster = PatchTSMixerForecaster(
        model_name="ibm-granite/granite-timeseries-patchtsmixer",
        context_length=200,
        prediction_length=60
    )
    
    # Preparar dados
    target_cols = ['base', 'high_corr', 'med_corr', 'low_corr', 'independent']
    multivariate_data, used_cols = forecaster.prepare_multivariate_data(
        data,
        target_cols=target_cols,
        time_col='timestamp'
    )
    
    # Forecast
    predictions, actual = forecaster.forecast(multivariate_data)
    
    # Análise das correlações nas previsões
    print("\\nCorrelações nas previsões:")
    pred_correlations = np.corrcoef(predictions.T)
    pred_corr_df = pd.DataFrame(
        pred_correlations,
        index=used_cols,
        columns=used_cols
    )
    print(pred_corr_df.round(3))
    
    # Comparar correlações
    print("\\nComparação correlações (Original vs Previsões):")
    print(f"{'Par':<20} {'Original':<10} {'Previsões':<10} {'Diferença':<10}")
    print("-" * 50)
    
    for i in range(len(used_cols)):
        for j in range(i+1, len(used_cols)):
            orig_corr = correlation_matrix.iloc[i, j]
            pred_corr = pred_correlations[i, j]
            diff = abs(orig_corr - pred_corr)
            
            pair_name = f"{used_cols[i]}-{used_cols[j]}"
            print(f"{pair_name:<20} {orig_corr:<10.3f} {pred_corr:<10.3f} {diff:<10.3f}")
    
    return predictions, actual, correlation_matrix

def example_patch_mixing_analysis():
    """
    Exemplo de análise do mecanismo de patch mixing
    """
    print("\\n=== Exemplo 4: Análise do Patch Mixing ===")
    
    # Criar dados com padrões em diferentes escalas
    np.random.seed(42)
    n_timesteps = 512  # Potência de 2 para melhor análise de patches
    dates = pd.date_range('2020-01-01', periods=n_timesteps, freq='H')
    
    # Canal com padrões em múltiplas escalas
    # Escala 1: Padrão horário (24h)
    hourly = 10 * np.sin(2 * np.pi * np.arange(n_timesteps) / 24)
    
    # Escala 2: Padrão diário (7d)
    daily = 15 * np.sin(2 * np.pi * np.arange(n_timesteps) / (24 * 7))
    
    # Escala 3: Padrão mensal
    monthly = 20 * np.sin(2 * np.pi * np.arange(n_timesteps) / (24 * 30))
    
    # Escala 4: Tendência
    trend = 0.1 * np.arange(n_timesteps)
    
    # Ruído
    noise = np.random.normal(0, 2, n_timesteps)
    
    # Combinar
    complex_channel = 100 + hourly + daily + monthly + trend + noise
    
    # Canal mais simples para comparação
    simple_channel = 50 + 5 * np.sin(2 * np.pi * np.arange(n_timesteps) / 100) + np.random.normal(0, 1, n_timesteps)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'complex': complex_channel,
        'simple': simple_channel
    })
    
    print("Dados com padrões em múltiplas escalas:")
    print(data.head())
    print(f"- Complex channel: min={complex_channel.min():.2f}, max={complex_channel.max():.2f}")
    print(f"- Simple channel: min={simple_channel.min():.2f}, max={simple_channel.max():.2f}")
    
    # Inicializar forecaster
    forecaster = PatchTSMixerForecaster(
        model_name="ibm-granite/granite-timeseries-patchtsmixer",
        context_length=400,
        prediction_length=96
    )
    
    # Preparar dados
    target_cols = ['complex', 'simple']
    multivariate_data, used_cols = forecaster.prepare_multivariate_data(
        data,
        target_cols=target_cols,
        time_col='timestamp'
    )
    
    # Testar diferentes configurações de patch
    patch_configs = [
        {'length': 8, 'stride': 4},
        {'length': 16, 'stride': 8},
        {'length': 32, 'stride': 16}
    ]
    
    results = {}
    
    for config in patch_configs:
        print(f"\\nTestando patches: length={config['length']}, stride={config['stride']}")
        
        # Criar patches
        patches = forecaster.create_patches(
            multivariate_data,
            patch_length=config['length'],
            patch_stride=config['stride']
        )
        
        print(f"  - Patches criados: {patches.shape}")
        
        # Forecast
        predictions, actual = forecaster.forecast(
            multivariate_data,
            patch_length=config['length'],
            patch_stride=config['stride']
        )
        
        # Métricas
        channel_results = {}
        for i, channel in enumerate(used_cols):
            pred = predictions[:, i]
            target = actual[:, i]
            
            mae = np.mean(np.abs(pred - target))
            rmse = np.sqrt(np.mean((pred - target)**2))
            
            channel_results[channel] = {'mae': mae, 'rmse': rmse}
        
        results[f"patch_{config['length']}_{config['stride']}"] = {
            'channel_results': channel_results,
            'n_patches': patches.shape[0]
        }
        
        print(f"  - Complex: MAE={channel_results['complex']['mae']:.2f}, RMSE={channel_results['complex']['rmse']:.2f}")
        print(f"  - Simple: MAE={channel_results['simple']['mae']:.2f}, RMSE={channel_results['simple']['rmse']:.2f}")
    
    # Melhor configuração
    best_config = min(
        results.keys(),
        key=lambda x: sum(results[x]['channel_results'][ch]['mae'] for ch in used_cols)
    )
    
    print(f"\\nMelhor configuração de patch: {best_config}")
    for channel in used_cols:
        print(f"  - {channel}: MAE={results[best_config]['channel_results'][channel]['mae']:.2f}")
    
    return results

if __name__ == "__main__":
    print("=== IBM PatchTSMixer Time Series Forecasting ===\\n")
    
    try:
        # Exemplo 1: Forecasting multivariado básico
        results1 = example_basic_multivariate_forecasting()
        
        # Exemplo 2: Integração de covariáveis
        pred2, actual2 = example_covariate_integration()
        
        # Exemplo 3: Análise de correlação
        pred3, actual3, corr3 = example_channel_correlation_analysis()
        
        # Exemplo 4: Análise do patch mixing
        results4 = example_patch_mixing_analysis()
        
        print("\\n=== Resumo ===")
        print("✓ Exemplo 1: Forecasting multivariado básico")
        print("✓ Exemplo 2: Integração de covariáveis dinâmicas")
        print("✓ Exemplo 3: Análise de correlação entre canais")
        print("✓ Exemplo 4: Análise do mecanismo de patch mixing")
        print("\\nPatchTSMixer é ideal para:")
        print("- Forecasting multivariado com correlações entre canais")
        print("- Modelos leves e eficientes (MLP-Mixer architecture)")
        print("- Mistura entre patches, canais e features ocultas")
        print("- Patterns complexos em múltiplas escalas temporais")
        print("- Eficiência computacional superior aos Transformers")
        
    except Exception as e:
        print(f"Erro durante execução: {e}")
        print("\\nPara usar PatchTSMixer, certifique-se de que transformers está instalado:")
        print("pip install transformers")
        print("\\nE que o modelo ibm-granite/granite-timeseries-patchtsmixer está disponível no Hugging Face")