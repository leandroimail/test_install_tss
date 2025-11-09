"""
IBM TTM (TinyTimeMixers): Compact Pre-trained Models for Multivariate Time Series Forecasting

TTM são modelos compactos (< 1M parâmetros) para forecasting multivariado.
Suporte para Exogenous Infusion (Dynamic Covariates) e Static Categorical Data.
"""

import os
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

# Instalar granite-tsfm se necessário
try:
    from tsfm_public import TinyTimeMixerForPrediction
    from tsfm_public.toolkit.get_model import get_model
    from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
except ImportError:
    print("Instalando granite-tsfm...")
    os.system("pip install git+https://github.com/ibm-granite/granite-tsfm.git")
    from tsfm_public import TinyTimeMixerForPrediction
    from tsfm_public.toolkit.get_model import get_model
    from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

class TTMForecaster:
    def __init__(self, model_name="ibm-granite/granite-timeseries-ttm-r2", 
                 device="auto", context_length=512, prediction_length=96):
        """
        Inicializa o forecaster TTM
        
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
        
        print(f"Carregando modelo TTM {model_name} no device {self.device}...")
        
        # Carregar modelo
        self.model = TinyTimeMixerForPrediction.from_pretrained(
            model_name,
            revision="main"
        ).to(self.device)
        
        # Modo de avaliação
        self.model.eval()
        
        print("TTM model carregado com sucesso!")
        print(f"- Contexto: {context_length}")
        print(f"- Predição: {prediction_length}")
    
    def auto_select_model(self, data_freq="D", prefer_longer_context=True):
        """
        Auto-seleciona modelo TTM baseado nos dados
        
        Args:
            data_freq: Frequência dos dados (ex: '10min', 'H', 'D', 'W')
            prefer_longer_context: Se prefere contexto mais longo
        """
        print(f"Auto-selecionando modelo TTM...")
        print(f"- Frequência: {data_freq}")
        print(f"- Contexto preferido: {prefer_longer_context}")
        
        # Usar utilitário de auto-seleção
        model_key = get_model(
            model_path=self.model_name,
            model_name='ttm',
            freq=data_freq,
            prefer_longer_context=prefer_longer_context
        )
        
        print(f"Modelo selecionado: {model_key}")
        return model_key
    
    def preprocess_data(self, data, target_cols, time_col, 
                       exogenous_cols=None, static_categorical_cols=None,
                       scale_data=True, fit_preprocessor=True):
        """
        Pré-processa dados para TTM
        
        Args:
            data: DataFrame com os dados
            target_cols: Lista de colunas alvo
            time_col: Nome da coluna de tempo
            exogenous_cols: Lista de colunas exógenas
            static_categorical_cols: Lista de colunas categóricas estáticas
            scale_data: Se deve normalizar os dados
            fit_preprocessor: Se deve ajustar o pré-processador
        """
        # Inicializar preprocessor
        self.preprocessor = TimeSeriesPreprocessor(
            freq=pd.infer_freq(data[time_col]),
            target_cols=target_cols,
            past_exogenous_cols=exogenous_cols or [],
            future_exogenous_cols=exogenous_cols or [],
            static_categorical_cols=static_categorical_cols or []
        )
        
        print(f"Pré-processando dados:")
        print(f"- Targets: {target_cols}")
        print(f"- Exógenas: {exogenous_cols}")
        print(f"- Categóricas estáticas: {static_categorical_cols}")
        print(f"- Normalização: {scale_data}")
        
        if fit_preprocessor:
            # Ajustar preprocessor
            preprocessed_data = self.preprocessor.fit_transform(data)
            print("Preprocessor ajustado e dados transformados")
        else:
            # Apenas transformar
            preprocessed_data = self.preprocessor.transform(data)
            print("Dados transformados (preprocessor já ajustado)")
        
        self.preprocessed_data = preprocessed_data
        
        return preprocessed_data
    
    def prepare_context_data(self, preprocessed_data, context_length=None):
        """
        Prepara dados de contexto para TTM
        
        Args:
            preprocessed_data: Dados pré-processados
            context_length: Comprimento do contexto (se None, usa o do modelo)
        """
        if context_length is None:
            context_length = self.context_length
        
        # Extrair contexto
        context_data = []
        target_data = []
        
        for entry in preprocessed_data:
            # Pegar contexto
            if len(entry['target']) >= context_length + self.prediction_length:
                context_series = entry['target'][-context_length:]
                target_series = entry['target'][-self.prediction_length:]
                
                context_data.append(context_series)
                target_data.append(target_series)
        
        print(f"Dados de contexto preparados:")
        print(f"- Séries: {len(context_data)}")
        print(f"- Contexto: {context_length}")
        print(f"- Predição: {self.prediction_length}")
        
        return context_data, target_data
    
    def forecast(self, preprocessed_data, exogenous_data=None, return_samples=False):
        """
        Gera forecast usando TTM
        
        Args:
            preprocessed_data: Dados pré-processados
            exogenous_data: Dados exógenas (opcional)
            return_samples: Se deve retornar samples
            
        Returns:
            Previsões
        """
        print(f"Gerando forecast com TTM...")
        
        # Preparar dados de entrada
        context_data, target_data = self.prepare_context_data(preprocessed_data)
        
        if not context_data:
            raise ValueError("Dados insuficientes para forecast")
        
        # Usar primeira série para demonstração
        input_batch = {
            'past_values': torch.tensor(
                context_data[0].values, 
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0),  # (1, context_length)
            'past_time_features': torch.tensor(
                context_data[0].time_features,
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)  # (1, context_length, n_features)
        }
        
        # Adicionar características exógenas se fornecidas
        if exogenous_data and 'past_exogenous' in exogenous_data[0]:
            input_batch['past_exogenous'] = torch.tensor(
                exogenous_data[0]['past_exogenous'],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)
        
        if exogenous_data and 'future_exogenous' in exogenous_data[0]:
            input_batch['future_exogenous'] = torch.tensor(
                exogenous_data[0]['future_exogenous'],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)
        
        print(f"Input batch shapes:")
        for key, value in input_batch.items():
            print(f"- {key}: {value.shape}")
        
        # Fazer forecast
        with torch.no_grad():
            outputs = self.model.generate(
                **input_batch,
                max_new_tokens=self.prediction_length,
                num_return_sequences=1,
                do_sample=False
            )
        
        # Extrair previsões
        predictions = outputs.sequences.squeeze(0).cpu().numpy()
        
        print(f"Forecast shape: {predictions.shape}")
        print("Forecast gerado com sucesso!")
        
        if return_samples:
            return {
                'predictions': predictions,
                'target': target_data[0] if target_data else None,
                'context': context_data[0] if context_data else None
            }
        
        return predictions

def create_ttm_sample_data(n_timesteps=500, n_exogenous=3, n_series=1):
    """
    Cria dados sintéticos para TTM com covariáveis exógenas
    """
    np.random.seed(42)
    data_list = []
    
    for series_id in range(n_series):
        # Gerar timestamps
        dates = pd.date_range('2020-01-01', periods=n_timesteps, freq='D')
        
        # Série principal (vendas)
        trend = np.linspace(100, 150, n_timesteps)
        seasonal = 20 * np.sin(2 * np.pi * np.arange(n_timesteps) / 365.25)
        noise = np.random.normal(0, 3, n_timesteps)
        sales = trend + seasonal + noise
        
        # Covariáveis exógenas dinâmicas
        # Preço (influência negativa nas vendas)
        price = 50 + 5 * np.sin(2 * np.pi * np.arange(n_timesteps) / 30) + np.random.normal(0, 1, n_timesteps)
        price_effect = -0.5 * (price - 50)  # Efeito nas vendas
        
        # Temperatura (influência nas vendas)
        temp_base = 20 + 10 * np.sin(2 * np.pi * np.arange(n_timesteps) / 365.25)
        temperature = temp_base + np.random.normal(0, 2, n_timesteps)
        temp_effect = 0.3 * (temperature - 20)  # Efeito nas vendas
        
        # Promoção (binária)
        promotion = np.random.binomial(1, 0.2, n_timesteps)
        promo_effect = 20 * promotion  # Efeito positivo
        
        # Catastrófico (eventos especiais)
        catastrophic = np.random.binomial(1, 0.05, n_timesteps)
        catastrophic_effect = -50 * catastrophic  # Efeito negativo
        
        # Calcular vendas com efeitos
        final_sales = sales + price_effect + temp_effect + promo_effect + catastrophic_effect + np.random.normal(0, 1, n_timesteps)
        
        # Métricas relacionadas
        conversion_rate = 0.05 + 0.01 * promotion + np.random.normal(0, 0.005, n_timesteps)
        customer_acquisition = 100 + 10 * promotion + np.random.normal(0, 5, n_timesteps)
        
        # Coluna categórica estática
        store_type = 'premium' if series_id == 0 else 'standard'
        
        for i, date in enumerate(dates):
            data_list.append({
                'timestamp': date,
                'sales': final_sales[i],
                'conversion_rate': conversion_rate[i],
                'customer_acquisition': customer_acquisition[i],
                'price': price[i],
                'temperature': temperature[i],
                'promotion': promotion[i],
                'catastrophic_event': catastrophic[i],
                'store_type': store_type,
                'store_id': f'store_{series_id}'
            })
    
    return pd.DataFrame(data_list)

def example_basic_ttm_forecasting():
    """
    Exemplo básico de forecasting com TTM
    """
    print("=== Exemplo 1: Forecasting Básico TTM ===")
    
    # Criar dados sintéticos
    data = create_ttm_sample_data(n_timesteps=300, n_exogenous=3, n_series=1)
    
    print("Dados criados:")
    print(data.head())
    print(f"Shape: {data.shape}")
    print(f"Colunas: {list(data.columns)}")
    
    # Auto-selecionar modelo
    forecaster = TTMForecaster(
        model_name="ibm-granite/granite-timeseries-ttm-r2",
        context_length=512,
        prediction_length=48
    )
    
    # Auto-seleção
    selected_model = forecaster.auto_select_model(data_freq="D")
    
    # Pré-processar dados
    target_cols = ['sales', 'conversion_rate', 'customer_acquisition']
    exogenous_cols = ['price', 'temperature', 'promotion']
    
    preprocessed_data = forecaster.preprocess_data(
        data=data,
        target_cols=target_cols,
        time_col='timestamp',
        exogenous_cols=exogenous_cols,
        static_categorical_cols=['store_type'],
        scale_data=True,
        fit_preprocessor=True
    )
    
    print(f"\\nDados pré-processados:")
    print(f"- Entradas: {len(preprocessed_data)}")
    print(f"- Primeira entrada shape: {preprocessed_data[0]['target'].shape}")
    
    # Fazer forecast
    results = forecaster.forecast(
        preprocessed_data,
        return_samples=True
    )
    
    print(f"\\nResultados do forecast:")
    print(f"- Predictions shape: {results['predictions'].shape}")
    print(f"- Target shape: {results['target'].shape}")
    
    # Mostrar previsões
    for i, col in enumerate(target_cols):
        pred_mean = np.mean(results['predictions'][:, i])
        pred_std = np.std(results['predictions'][:, i])
        
        print(f"\\n{col}:")
        print(f"  - Média previsão: {pred_mean:.2f}")
        print(f"  - Desvio padrão: {pred_std:.2f}")
        print(f"  - Primeiras 5: {results['predictions'][:5, i]}")
    
    return results

def example_ttm_with_exogenous_variables():
    """
    Exemplo com variáveis exógenas
    """
    print("\\n=== Exemplo 2: TTM com Variáveis Exógenas ===")
    
    # Criar dados com séries mais complexas
    data = create_ttm_sample_data(n_timesteps=400, n_exogenous=4, n_series=2)
    
    print("Dados com variáveis exógenas:")
    print(data.head())
    print(f"Shape: {data.shape}")
    print(f"Séries: {data['store_id'].unique()}")
    
    # Inicializar forecaster
    forecaster = TTMForecaster(
        model_name="ibm-granite/granite-timeseries-ttm-r2",
        context_length=256,
        prediction_length=30
    )
    
    # Pré-processar por série
    all_results = {}
    
    for store_id in data['store_id'].unique():
        print(f"\\nProcessando {store_id}:")
        store_data = data[data['store_id'] == store_id]
        
        # Definir colunas
        target_cols = ['sales']
        exogenous_cols = ['price', 'temperature', 'promotion', 'catastrophic_event']
        categorical_cols = ['store_type']
        
        # Pré-processar
        preprocessed_data = forecaster.preprocess_data(
            data=store_data,
            target_cols=target_cols,
            time_col='timestamp',
            exogenous_cols=exogenous_cols,
            static_categorical_cols=categorical_cols,
            scale_data=True,
            fit_preprocessor=True
        )
        
        # Fazer forecast
        results = forecaster.forecast(
            preprocessed_data,
            return_samples=True
        )
        
        all_results[store_id] = results
        
        print(f"- Predictions shape: {results['predictions'].shape}")
        print(f"- Primeira previsão: {results['predictions'][0, 0]:.2f}")
    
    # Comparar previsões entre lojas
    print(f"\\nComparação entre lojas:")
    for store_id, results in all_results.items():
        mean_prediction = np.mean(results['predictions'])
        print(f"- {store_id}: {mean_prediction:.2f}")
    
    return all_results

def example_ttm_channel_mixing():
    """
    Exemplo usando channel mixing para capturar correlações
    """
    print("\\n=== Exemplo 3: TTM com Channel Mixing ===")
    
    # Criar dados com métricas fortemente correlacionadas
    np.random.seed(42)
    n_timesteps = 300
    dates = pd.date_range('2020-01-01', periods=n_timesteps, freq='D')
    
    # Métricas de e-commerce correlacionadas
    # Tráfego
    traffic_base = 1000 + 200 * np.sin(2 * np.pi * np.arange(n_timesteps) / 7)  # Weekly pattern
    traffic = traffic_base + np.random.normal(0, 50, n_timesteps)
    
    # Conversão (relacionada ao tráfego)
    conversion_rate = 0.02 + 0.001 * (traffic - 1000) / 100 + np.random.normal(0, 0.001, n_timesteps)
    conversion_rate = np.clip(conversion_rate, 0.001, 0.1)
    
    # Receita (tráfego × conversão × ticket médio)
    avg_ticket = 50 + 5 * np.sin(2 * np.pi * np.arange(n_timesteps) / 30)
    revenue = traffic * conversion_rate * avg_ticket
    
    # Custos (proporcionais ao tráfego)
    costs = 0.3 * traffic + 100 + np.random.normal(0, 20, n_timesteps)
    
    # Margem
    margin = revenue - costs
    
    # Covariáveis exógenas
    price_index = 100 + 5 * np.sin(2 * np.pi * np.arange(n_timesteps) / 90) + np.random.normal(0, 2, n_timesteps)
    promotion = np.random.binomial(1, 0.15, n_timesteps)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'traffic': traffic,
        'conversion_rate': conversion_rate,
        'revenue': revenue,
        'costs': costs,
        'margin': margin,
        'price_index': price_index,
        'promotion': promotion,
        'channel': 'ecommerce'
    })
    
    print("Dados de e-commerce:")
    print(data.head())
    print(f"\\nCorrelações:")
    target_cols = ['traffic', 'conversion_rate', 'revenue', 'costs', 'margin']
    correlation_matrix = data[target_cols].corr()
    print(correlation_matrix.round(3))
    
    # Inicializar forecaster
    forecaster = TTMForecaster(
        model_name="ibm-granite/granite-timeseries-ttm-r2",
        context_length=512,
        prediction_length=60
    )
    
    # Pré-processar
    preprocessed_data = forecaster.preprocess_data(
        data=data,
        target_cols=target_cols,
        time_col='timestamp',
        exogenous_cols=['price_index', 'promotion'],
        static_categorical_cols=['channel'],
        scale_data=True,
        fit_preprocessor=True
    )
    
    # Fazer forecast
    results = forecaster.forecast(
        preprocessed_data,
        return_samples=True
    )
    
    print(f"\\nForecast results:")
    print(f"- Shape: {results['predictions'].shape}")
    
    # Análise de correlações nas previsões
    pred_correlations = np.corrcoef(results['predictions'].T)
    
    print(f"\\nCorrelações nas previsões:")
    pred_corr_df = pd.DataFrame(
        pred_correlations,
        index=target_cols,
        columns=target_cols
    )
    print(pred_corr_df.round(3))
    
    # Comparar com correlações originais
    print(f"\\nComparação correlações (Original vs Previsões):")
    for i in range(len(target_cols)):
        for j in range(i+1, len(target_cols)):
            orig_corr = correlation_matrix.iloc[i, j]
            pred_corr = pred_correlations[i, j]
            print(f"- {target_cols[i]} vs {target_cols[j]}: {orig_corr:.3f} vs {pred_corr:.3f}")
    
    return results

def example_ttm_fine_tuning():
    """
    Exemplo de fine-tuning do TTM
    """
    print("\\n=== Exemplo 4: TTM Fine-tuning ===")
    
    # Criar dados de treino e teste
    train_data = create_ttm_sample_data(n_timesteps=300, n_exogenous=3, n_series=1)
    test_data = create_ttm_sample_data(n_timesteps=150, n_exogenous=3, n_series=1)
    
    print("Dados para fine-tuning:")
    print(f"- Treino: {train_data.shape}")
    print(f"- Teste: {test_data.shape}")
    
    # Inicializar forecaster
    forecaster = TTMForecaster(
        model_name="ibm-granite/granite-timeseries-ttm-r2",
        context_length=256,
        prediction_length=30
    )
    
    # Pré-processar dados de treino
    target_cols = ['sales']
    exogenous_cols = ['price', 'temperature', 'promotion']
    
    train_preprocessed = forecaster.preprocess_data(
        data=train_data,
        target_cols=target_cols,
        time_col='timestamp',
        exogenous_cols=exogenous_cols,
        static_categorical_cols=['store_type'],
        scale_data=True,
        fit_preprocessor=True
    )
    
    # Preparar dados de treino
    train_context, train_targets = forecaster.prepare_context_data(train_preprocessed)
    
    print(f"\\nDados de treino preparados:")
    print(f"- Contexto: {len(train_context)} séries")
    print(f"- Shape contexto: {train_context[0].shape}")
    print(f"- Shape targets: {train_targets[0].shape}")
    
    # Simular fine-tuning (em implementação real, seria com otimizador)
    print(f"\\nSimulando fine-tuning...")
    print(f"Em implementação real, seria necessário:")
    print(f"- Configurar otimizador (Adam, learning rate, etc.)")
    print(f"- Loop de treinamento com loss function")
    print(f"- Validação em dados de teste")
    print(f"- Here usamos apenas o modelo pré-treinado para demo")
    
    # Fazer previsões com dados de teste
    test_preprocessed = forecaster.preprocess_data(
        data=test_data,
        target_cols=target_cols,
        time_col='timestamp',
        exogenous_cols=exogenous_cols,
        static_categorical_cols=['store_type'],
        scale_data=False,  # Usar scaler já ajustado
        fit_preprocessor=False
    )
    
    test_results = forecaster.forecast(
        test_preprocessed,
        return_samples=True
    )
    
    print(f"\\nResultados de teste:")
    print(f"- Predictions shape: {test_results['predictions'].shape}")
    print(f"- Target shape: {test_results['target'].shape}")
    
    # Avaliar performance
    if test_results['target'] is not None:
        mae = np.mean(np.abs(test_results['predictions'][:, 0] - test_results['target'].values))
        rmse = np.sqrt(np.mean((test_results['predictions'][:, 0] - test_results['target'].values)**2))
        
        print(f"- MAE: {mae:.2f}")
        print(f"- RMSE: {rmse:.2f}")
    
    return test_results

if __name__ == "__main__":
    print("=== IBM TTM (TinyTimeMixers) Time Series Forecasting ===\\n")
    
    try:
        # Exemplo 1: Forecasting básico
        results1 = example_basic_ttm_forecasting()
        
        # Exemplo 2: Com variáveis exógenas
        results2 = example_ttm_with_exogenous_variables()
        
        # Exemplo 3: Channel mixing
        results3 = example_ttm_channel_mixing()
        
        # Exemplo 4: Fine-tuning
        results4 = example_ttm_fine_tuning()
        
        print("\\n=== Resumo ===")
        print("✓ Exemplo 1: Forecasting básico com TTM")
        print("✓ Exemplo 2: TTM com variáveis exógenas")
        print("✓ Exemplo 3: Channel mixing para correlações")
        print("✓ Exemplo 4: Fine-tuning simulation")
        print("\\nTTM é ideal para:")
        print("- Modelos compactos (< 1M parâmetros)")
        print("- Forecasting multivariado com correlações")
        print("- Integração de variáveis exógenas")
        print("- Fine-tuning rápido (minutos)")
        print("- Zero-shot e few-shot learning")
        
    except Exception as e:
        print(f"Erro durante execução: {e}")
        print("\\nPara usar TTM, certifique-se de que granite-tsfm está instalado:")
        print("pip install git+https://github.com/ibm-granite/granite-tsfm.git")