"""
Salesforce Moirai-2.0: Zero-Shot Time Series Forecasting with Dynamic Covariates

Moirai-2.0 é um modelo de foundation transformer para forecasting universal.
Suporte para covariáveis dinâmicas através dos parâmetros feat_dynamic_real_dim 
e past_feat_dynamic_real_dim.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

# Instalar uni2ts se necessário
# os.system("pip install git+https://github.com/SalesforceAIResearch/uni2ts.git")

try:
    from uni2ts.model.moirai import Moirai2Forecast
    from uni2ts.data.builder import GluonTSDataset
    from uni2ts.eval_util.metrics import MASE, CRPS
    from uni2ts.data.util.dataset import PandasDataset
    from gluonts.dataset.split import DateOffsetInputSplitter
    from gluonts.itertools import Iterable
    from gluonts.evaluation import make_evaluation_predictions
    from gluonts.evaluation.backtest import backtest_metrics
    from gluonts.model.forecast import QuantileForecast
except ImportError:
    print("Instalando uni2ts...")
    os.system("pip install git+https://github.com/SalesforceAIResearch/uni2ts.git")
    from uni2ts.model.moirai import Moirai2Forecast
    from uni2ts.data.builder import GluonTSDataset
    from uni2ts.eval_util.metrics import MASE, CRPS
    from uni2ts.data.util.dataset import PandasDataset
    from gluonts.dataset.split import DateOffsetInputSplitter
    from gluonts.itertools import Iterable
    from gluonts.evaluation import make_evaluation_predictions
    from gluonts.evaluation.backtest import backtest_metrics
    from gluonts.model.forecast import QuantileForecast

class Moirai2Forecaster:
    def __init__(self, model_name="Salesforce/moirai-2.0-R-small", 
                 patch_size=64, context_length=512, prediction_length=96):
        """
        Inicializa o forecaster Moirai-2.0
        
        Args:
            model_name: Nome do modelo no Hugging Face
            patch_size: Tamanho do patch (auto, 8, 16, 32, 64, 128)
            context_length: Comprimento do contexto
            prediction_length: Comprimento da previsão
        """
        self.model_name = model_name
        self.patch_size = patch_size
        self.context_length = context_length
        self.prediction_length = prediction_length
        
        print(f"Carregando modelo {model_name}...")
        self.model = Moirai2Forecast.load_from_checkpoint(
            model_name,
            patch_size=patch_size,
            context_length=context_length,
            prediction_length=prediction_length,
        )
        print("Modelo carregado com sucesso!")
    
    def create_dataset(self, data, target_cols, time_col, id_col=None,
                      past_covariates=None, future_covariates=None):
        """
        Cria dataset para Moirai-2.0
        
        Args:
            data: DataFrame com os dados
            target_cols: Lista de colunas alvo
            time_col: Nome da coluna de tempo
            id_col: Nome da coluna de identificação
            past_covariates: Lista de covariáveis passadas
            future_covariates: Lista de covariáveis futuras
        """
        # Preparar colunas
        feat_dynamic_real_cols = []
        past_feat_dynamic_real_cols = []
        
        if future_covariates:
            feat_dynamic_real_cols.extend(future_covariates)
        
        if past_covariates:
            past_feat_dynamic_real_cols.extend(past_covariates)
        
        # Criar dataset GluonTS
        self.dataset = PandasDataset(
            dataframe=data,
            target=target_cols if isinstance(target_cols, list) else [target_cols],
            timestamp=time_col,
            id=id_col,
            feat_dynamic_real=feat_dynamic_real_cols if feat_dynamic_real_cols else None,
            past_feat_dynamic_real=past_feat_dynamic_real_cols if past_feat_dynamic_real_cols else None,
        )
        
        print(f"Dataset criado:")
        print(f"- Targets: {target_cols}")
        print(f"- Covariáveis futuras: {len(feat_dynamic_real_cols)}")
        print(f"- Covariáveis passadas: {len(past_feat_dynamic_real_cols)}")
    
    def split_data(self, split_offset="P1D"):
        """
        Divide os dados para treino e teste
        
        Args:
            split_offset: Offset para split (ex: "P1D" = 1 dia)
        """
        self.splitter = DateOffsetInputSplitter(offset=split_offset)
        train_data, test_data = self.splitter.split(self.dataset)
        
        print(f"Dados divididos:")
        print(f"- Treino: {len(list(train_data))} séries")
        print(f"- Teste: {len(list(test_data))} séries")
        
        return train_data, test_data
    
    def create_predictor(self, batch_size=32):
        """
        Cria o predictor para fazer previsões
        """
        self.predictor = self.model.create_predictor(
            batch_size=batch_size,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        print("Predictor criado!")
    
    def forecast(self, test_data):
        """
        Gera forecast usando Moirai-2.0
        
        Args:
            test_data: Dados de teste
            
        Returns:
            Previsões geradas
        """
        print("Gerando forecast...")
        
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_data,
            predictor=self.predictor,
            num_samples=100,
        )
        
        forecasts = list(forecast_it)
        print(f"Forecast gerado para {len(forecasts)} séries")
        
        return forecasts
    
    def evaluate(self, test_data, forecasts):
        """
        Avalia as previsões
        """
        print("Avaliando forecast...")
        
        ts_it = make_evaluation_predictions(
            dataset=test_data,
            predictor=self.predictor,
            num_samples=100,
        )
        
        # Calcular métricas
        metrics = backtest_metrics(
            test_dataset=test_data,
            forecaster=self.predictor,
            metrics=[MASE(), CRPS()],
        )
        
        print("Métricas de avaliação:")
        for metric_name, value in metrics:
            print(f"- {metric_name}: {value:.4f}")
        
        return metrics

def example_basic_forecasting():
    """
    Exemplo básico de forecasting com Moirai-2.0
    """
    # Criar dados sintéticos
    np.random.seed(42)
    n_timesteps = 500
    dates = pd.date_range('2020-01-01', periods=n_timesteps, freq='D')
    
    # Série temporal com tendência e sazonalidade
    trend = np.linspace(100, 150, n_timesteps)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n_timesteps) / 365.25)
    noise = np.random.normal(0, 5, n_timesteps)
    sales = trend + seasonal + noise
    
    # Criar DataFrame
    data = pd.DataFrame({
        'date': dates,
        'target': sales,
        'id': 'series_1'
    })
    
    print("Dados criados:")
    print(data.head())
    
    # Inicializar forecaster
    forecaster = Moirai2Forecaster(
        model_name="Salesforce/moirai-2.0-R-small",
        patch_size=64,
        context_length=512,
        prediction_length=96
    )
    
    # Criar dataset
    forecaster.create_dataset(
        data=data,
        target_cols='target',
        time_col='date',
        id_col='id'
    )
    
    # Dividir dados
    train_data, test_data = forecaster.split_data(split_offset="P30D")
    
    # Criar predictor
    forecaster.create_predictor(batch_size=8)
    
    # Gerar forecast
    forecasts = forecaster.forecast(test_data)
    
    # Mostrar resultado
    if forecasts:
        print(f"\\nPrimeira previsão:")
        print(f"- Shape: {forecasts[0].samples.shape}")
        print(f"- Quantis: {forecasts[0].quantile(0.5)}")
        print(f"- Mean: {forecasts[0].mean}")
    
    return forecasts

def example_with_dynamic_covariates():
    """
    Exemplo de forecasting com covariáveis dinâmicas
    """
    # Criar dados sintéticos com covariáveis
    np.random.seed(42)
    n_timesteps = 800
    dates = pd.date_range('2020-01-01', periods=n_timesteps, freq='D')
    
    # Série principal (vendas)
    trend = np.linspace(100, 160, n_timesteps)
    seasonal = 15 * np.sin(2 * np.pi * np.arange(n_timesteps) / 365.25)
    noise = np.random.normal(0, 4, n_timesteps)
    sales = trend + seasonal + noise
    
    # Covariáveis dinâmicas
    # Preço (covariável futura conhecida)
    price = 50 + 5 * np.sin(2 * np.pi * np.arange(n_timesteps) / 30) + np.random.normal(0, 1, n_timesteps)
    
    # Temperatura (covariável passada)
    temp_base = 20 + 10 * np.sin(2 * np.pi * np.arange(n_timesteps) / 365.25)
    temp_seasonal = 5 * np.sin(2 * np.pi * np.arange(n_timesteps) / 7)
    temperature = temp_base + temp_seasonal + np.random.normal(0, 2, n_timesteps)
    
    # Promoção (covariável binária)
    promotion = np.random.binomial(1, 0.2, n_timesteps)
    
    # Criar DataFrame
    data = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'price': price,
        'temperature': temperature,
        'promotion': promotion,
        'id': 'retail_1'
    })
    
    print("Dados com covariáveis criados:")
    print(data.head())
    print(f"Shape: {data.shape}")
    
    # Inicializar forecaster
    forecaster = Moirai2Forecaster(
        model_name="Salesforce/moirai-2.0-R-small",
        patch_size=32,
        context_length=512,
        prediction_length=48
    )
    
    # Criar dataset com covariáveis
    forecaster.create_dataset(
        data=data,
        target_cols='sales',
        time_col='date',
        id_col='id',
        past_covariates=['temperature', 'promotion'],
        future_covariates=['price']
    )
    
    # Dividir dados
    train_data, test_data = forecaster.split_data(split_offset="P30D")
    
    # Criar predictor
    forecaster.create_predictor(batch_size=8)
    
    # Gerar forecast
    forecasts = forecaster.forecast(test_data)
    
    # Avaliar
    metrics = forecaster.evaluate(test_data, forecasts)
    
    return forecasts, metrics

def example_multivariate_with_covariates():
    """
    Exemplo de forecasting multivariado com covariáveis
    """
    # Criar dados com múltiplas séries
    np.random.seed(42)
    n_timesteps = 600
    dates = pd.date_range('2020-01-01', periods=n_timesteps, freq='D')
    
    data_list = []
    
    for series_id in ['store_A', 'store_B', 'store_C']:
        # Série principal
        trend = np.linspace(100, 130, n_timesteps) + np.random.normal(0, 5)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_timesteps) / 365.25)
        sales = trend + seasonal + np.random.normal(0, 3)
        
        # Covariáveis
        price = 50 + 3 * np.sin(2 * np.pi * np.arange(n_timesteps) / 30) + np.random.normal(0, 1)
        temperature = 20 + 8 * np.sin(2 * np.pi * np.arange(n_timesteps) / 365.25) + np.random.normal(0, 2)
        promotion = np.random.binomial(1, 0.15, n_timesteps)
        
        for i, date in enumerate(dates):
            data_list.append({
                'date': date,
                'sales': sales[i],
                'price': price[i],
                'temperature': temperature[i],
                'promotion': promotion[i],
                'id': series_id
            })
    
    data = pd.DataFrame(data_list)
    
    print("Dados multivariados criados:")
    print(data.head(10))
    print(f"Shape: {data.shape}")
    print(f"Séries únicas: {data['id'].unique()}")
    
    # Inicializar forecaster
    forecaster = Moirai2Forecaster(
        model_name="Salesforce/moirai-2.0-R-small",
        patch_size=32,
        context_length=256,
        prediction_length=30
    )
    
    # Criar dataset multivariado
    forecaster.create_dataset(
        data=data,
        target_cols=['sales'],
        time_col='date',
        id_col='id',
        past_covariates=['temperature', 'promotion'],
        future_covariates=['price']
    )
    
    # Dividir dados
    train_data, test_data = forecaster.split_data(split_offset="P14D")
    
    # Criar predictor
    forecaster.create_predictor(batch_size=16)
    
    # Gerar forecast
    forecasts = forecaster.forecast(test_data)
    
    print(f"\\nForecast multivariado:")
    print(f"Número de séries previstas: {len(forecasts)}")
    
    for i, forecast in enumerate(forecasts[:2]):  # Mostrar apenas as 2 primeiras
        print(f"Série {i+1}:")
        print(f"  - Shape: {forecast.samples.shape}")
        print(f"  - Mean: {forecast.mean:.2f}")
        print(f"  - Quantil 50%: {forecast.quantile(0.5):.2f}")
    
    return forecasts

if __name__ == "__main__":
    print("=== Salesforce Moirai-2.0 Time Series Forecasting ===\\n")
    
    # Exemplo 1: Forecasting básico
    print("Exemplo 1: Forecasting básico")
    forecast1 = example_basic_forecasting()
    
    print("\\n" + "="*60 + "\\n")
    
    # Exemplo 2: Com covariáveis dinâmicas
    print("Exemplo 2: Forecasting com covariáveis dinâmicas")
    forecast2, metrics2 = example_with_dynamic_covariates()
    
    print("\\n" + "="*60 + "\\n")
    
    # Exemplo 3: Multivariado com covariáveis
    print("Exemplo 3: Forecasting multivariado com covariáveis")
    forecast3 = example_multivariate_with_covariates()
    
    print("\\nTodos os exemplos concluídos!")