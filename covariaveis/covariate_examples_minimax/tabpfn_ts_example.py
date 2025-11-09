"""
TabPFN-TS: Zero-Shot Time Series Forecasting with Dynamic Covariates

TabPFN-TS é uma implementação para forecasting zero-shot que usa TabPFNv2.
Suporte para variáveis exógenas e feature engineering automático.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Instalar tabpfn-time-series se necessário
try:
    from tabpfn_time_series import TabPFNRegressor
    from tabpfn_time_series.preprocessing import TimeSeriesTransformer
except ImportError:
    print("Instalando tabpfn-time-series...")
    import os
    os.system("pip install tabpfn-time-series")
    from tabpfn_time_series import TabPFNRegressor
    from tabpfn_time_series.preprocessing import TimeSeriesTransformer

class TabPFNTSForecaster:
    def __init__(self, n_estimators=10, device='cpu', add_noise=False):
        """
        Inicializa o forecaster TabPFN-TS
        
        Args:
            n_estimators: Número de estimadores (padrão: 10)
            device: Dispositivo ('cpu', 'cuda')
            add_noise: Se deve adicionar ruído durante treinamento
        """
        self.n_estimators = n_estimators
        self.device = device
        self.add_noise = add_noise
        
        self.regressor = TabPFNRegressor(
            n_estimators=n_estimators,
            device=device,
            add_noise=add_noise
        )
        self.transformer = TimeSeriesTransformer()
        
        print(f"TabPFN-TS forecaster inicializado:")
        print(f"- Estimadores: {n_estimators}")
        print(f"- Dispositivo: {device}")
        print(f"- Adicionar ruído: {add_noise}")
    
    def prepare_data_with_covariates(self, data, target_col, time_col, 
                                   covariates=None, lookback_window=128,
                                   prediction_length=1):
        """
        Prepara dados com covariáveis para TabPFN-TS
        
        Args:
            data: DataFrame com os dados
            target_col: Nome da coluna alvo
            time_col: Nome da coluna de tempo
            covariates: Lista de nomes de covariáveis
            lookback_window: Janela de histórico
            prediction_length: Passos para prever
        """
        self.target_col = target_col
        self.time_col = time_col
        self.covariates = covariates or []
        self.lookback_window = lookback_window
        self.prediction_length = prediction_length
        
        # Ordenar por tempo
        data = data.sort_values(time_col).reset_index(drop=True)
        
        # Preparar features
        feature_cols = [target_col] + self.covariates
        feature_data = data[feature_cols].values
        
        print(f"Dados preparados:")
        print(f"- Shape original: {data.shape}")
        print(f"- Features: {len(feature_cols)} ({feature_cols})")
        print(f"- Lookback: {lookback_window}")
        print(f"- Predição: {prediction_length}")
        
        return feature_data
    
    def prepare_multivariate_data(self, data, target_cols, time_col, 
                                covariates=None, lookback_window=128):
        """
        Prepara dados multivariados
        
        Args:
            data: DataFrame com os dados
            target_cols: Lista de colunas alvo
            time_col: Nome da coluna de tempo
            covariates: Lista de nomes de covariáveis
            lookback_window: Janela de histórico
        """
        self.target_cols = target_cols if isinstance(target_cols, list) else [target_cols]
        self.time_col = time_col
        self.covariates = covariates or []
        self.lookback_window = lookback_window
        
        # Preparar features
        feature_cols = self.target_cols + self.covariates
        feature_data = data[feature_cols].values
        
        print(f"Dados multivariados preparados:")
        print(f"- Shape: {data.shape}")
        print(f"- Targets: {len(self.target_cols)} ({self.target_cols})")
        print(f"- Covariáveis: {len(self.covariates)} ({self.covariates})")
        
        return feature_data
    
    def fit(self, X, y=None):
        """
        Treina o modelo TabPFN-TS
        
        Args:
            X: Dados de entrada (série temporal com covariáveis)
            y: Variável alvo (se None, usa última coluna de X)
        """
        if y is None:
            # Usar última coluna como target se y não especificado
            y = X[:, -1]
            X = X[:, :-1]  # Remover última coluna
        
        print("Treinando TabPFN-TS...")
        print(f"- X shape: {X.shape}")
        print(f"- y shape: {y.shape}")
        
        self.regressor.fit(X, y)
        print("Treinamento concluído!")
    
    def predict(self, X, return_std=False):
        """
        Faz previsões
        
        Args:
            X: Dados de entrada
            return_std: Se deve retornar desvio padrão
            
        Returns:
            Previsões
        """
        print("Gerando previsões...")
        print(f"- X shape: {X.shape}")
        
        if return_std:
            predictions, std = self.regressor.predict(X, return_std=True)
            print(f"- Previsões shape: {predictions.shape}")
            print(f"- Std shape: {std.shape}")
            return predictions, std
        else:
            predictions = self.regressor.predict(X)
            print(f"- Previsões shape: {predictions.shape}")
            return predictions
    
    def forecast_with_rolling_window(self, data, prediction_length=1):
        """
        Forecast usando rolling window
        
        Args:
            data: Dados de entrada
            prediction_length: Passos para prever
            
        Returns:
            Array com previsões
        """
        n_samples = len(data)
        lookback = self.lookback_window
        
        predictions = []
        
        print(f"Forecast com rolling window:")
        print(f"- Total samples: {n_samples}")
        print(f"- Lookback: {lookback}")
        print(f"- Prediction length: {prediction_length}")
        
        # Gerar previsões para cada janela
        for i in range(lookback, n_samples, prediction_length):
            # Pegar janela atual
            X_window = data[i-lookback:i]
            
            # Prever próximo valor
            pred = self.regressor.predict(X_window.reshape(1, -1))
            predictions.append(pred[0])
            
            # Atualizar janela (rolling)
            if i + prediction_length < n_samples:
                # Deslocar janela para frente
                data[i:i+prediction_length] = pred[0]
        
        return np.array(predictions)

def create_sample_data(n_timesteps=500, n_covariates=3, n_series=1):
    """
    Cria dados sintéticos com covariáveis
    """
    np.random.seed(42)
    
    data_list = []
    
    for series_id in range(n_series):
        # Gerar timestamps
        dates = pd.date_range('2020-01-01', periods=n_timesteps, freq='D')
        
        # Série principal
        trend = np.linspace(100, 150, n_timesteps)
        seasonal = 20 * np.sin(2 * np.pi * np.arange(n_timesteps) / 365.25)
        noise = np.random.normal(0, 3, n_timesteps)
        sales = trend + seasonal + noise
        
        # Covariáveis dinâmicas
        price = 50 + 5 * np.sin(2 * np.pi * np.arange(n_timesteps) / 30) + np.random.normal(0, 1, n_timesteps)
        temperature = 20 + 10 * np.sin(2 * np.pi * np.arange(n_timesteps) / 365.25) + np.random.normal(0, 2, n_timesteps)
        promotion = np.random.binomial(1, 0.2, n_timesteps)
        
        for i, date in enumerate(dates):
            data_list.append({
                'date': date,
                'sales': sales[i],
                'price': price[i],
                'temperature': temperature[i],
                'promotion': promotion[i],
                'series_id': f'series_{series_id}'
            })
    
    return pd.DataFrame(data_list)

def example_basic_forecasting():
    """
    Exemplo básico de forecasting com TabPFN-TS
    """
    print("=== Exemplo 1: Forecasting Básico ===")
    
    # Criar dados sintéticos
    data = create_sample_data(n_timesteps=300)
    
    print("Dados criados:")
    print(data.head())
    print(f"Shape: {data.shape}")
    
    # Inicializar forecaster
    forecaster = TabPFNTSForecaster(
        n_estimators=5,
        device='cpu',
        add_noise=True
    )
    
    # Dividir dados
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"\\nDivisão dos dados:")
    print(f"- Treino: {len(train_data)} samples")
    print(f"- Teste: {len(test_data)} samples")
    
    # Preparar dados de treino
    X_train = forecaster.prepare_data_with_covariates(
        train_data,
        target_col='sales',
        time_col='date',
        covariates=['price', 'temperature', 'promotion'],
        lookback_window=64
    )
    
    # Treinar
    forecaster.fit(X_train)
    
    # Preparar dados de teste
    test_window = test_data[['sales', 'price', 'temperature', 'promotion']].values
    
    # Fazer previsões
    predictions = forecaster.predict(test_window)
    actual = test_data['sales'].values
    
    print(f"\\nResultados:")
    print(f"- Previsões shape: {predictions.shape}")
    print(f"- Valores reais shape: {actual.shape}")
    print(f"- MAE: {np.mean(np.abs(predictions - actual)):.2f}")
    print(f"- RMSE: {np.sqrt(np.mean((predictions - actual)**2)):.2f}")
    
    return predictions, actual

def example_multivariate_forecasting():
    """
    Exemplo de forecasting multivariado
    """
    print("\\n=== Exemplo 2: Forecasting Multivariado ===")
    
    # Criar dados com múltiplas séries
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=400, freq='D')
    
    data_list = []
    
    for store in ['store_A', 'store_B']:
        # Vendas do produto
        trend = np.linspace(100, 140, 400)
        seasonal = 15 * np.sin(2 * np.pi * np.arange(400) / 365.25)
        sales = trend + seasonal + np.random.normal(0, 3, 400)
        
        # Vendas de produto relacionado
        related_sales = 0.7 * sales + 0.3 * np.random.normal(100, 20, 400)
        
        # Covariáveis
        price = 50 + 3 * np.sin(2 * np.pi * np.arange(400) / 30) + np.random.normal(0, 1, 400)
        promotion = np.random.binomial(1, 0.15, 400)
        
        for i, date in enumerate(dates):
            data_list.append({
                'date': date,
                'main_sales': sales[i],
                'related_sales': related_sales[i],
                'price': price[i],
                'promotion': promotion[i],
                'store': store
            })
    
    data = pd.DataFrame(data_list)
    
    print("Dados multivariados:")
    print(data.head())
    print(f"Shape: {data.shape}")
    print(f"Lojas: {data['store'].unique()}")
    
    # Dividir dados
    train_size = int(0.75 * 400)  # 400 samples por loja
    train_data = data[data.index % 400 < train_size]
    test_data = data[data.index % 400 >= train_size]
    
    # Processar por loja
    all_predictions = {}
    
    for store in data['store'].unique():
        print(f"\\nProcessando loja: {store}")
        
        store_train = train_data[train_data['store'] == store]
        store_test = test_data[test_data['store'] == store]
        
        # Inicializar forecaster
        forecaster = TabPFNTSForecaster(
            n_estimators=5,
            device='cpu'
        )
        
        # Preparar dados (prever main_sales usando related_sales como covariável)
        X_train = forecaster.prepare_multivariate_data(
            store_train,
            target_cols=['main_sales'],
            time_col='date',
            covariates=['related_sales', 'price', 'promotion'],
            lookback_window=48
        )
        
        # Treinar
        forecaster.fit(X_train)
        
        # Prever
        X_test = store_test[['main_sales', 'related_sales', 'price', 'promotion']].values
        predictions = forecaster.predict(X_test)
        actual = store_test['main_sales'].values
        
        all_predictions[store] = {
            'predictions': predictions,
            'actual': actual,
            'mae': np.mean(np.abs(predictions - actual))
        }
        
        print(f"- Previsões: {predictions.shape}")
        print(f"- MAE: {all_predictions[store]['mae']:.2f}")
    
    return all_predictions

def example_rolling_forecast():
    """
    Exemplo de forecast com rolling window
    """
    print("\\n=== Exemplo 3: Rolling Forecast ===")
    
    # Criar dados sintéticos
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='H')
    
    # Série com sazonalidade intradiária
    hourly_pattern = 20 * np.sin(2 * np.pi * np.arange(200) / 24)
    daily_trend = np.linspace(100, 120, 200)
    noise = np.random.normal(0, 2, 200)
    target = daily_trend + hourly_pattern + noise
    
    # Covariáveis
    temperature = 22 + 3 * np.sin(2 * np.pi * np.arange(200) / 24) + np.random.normal(0, 1, 200)
    demand = 0.8 * target + 0.2 * np.random.normal(100, 10, 200)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'demand': target,
        'temperature': temperature,
        'external_demand': demand
    })
    
    print("Dados para rolling forecast:")
    print(data.head())
    print(f"Shape: {data.shape}")
    
    # Inicializar forecaster
    forecaster = TabPFNTSForecaster(
        n_estimators=3,
        device='cpu'
    )
    
    # Preparar dados
    lookback = 24  # 24 horas de histórico
    feature_data = forecaster.prepare_multivariate_data(
        data,
        target_cols=['demand'],
        time_col='timestamp',
        covariates=['temperature', 'external_demand'],
        lookback_window=lookback
    )
    
    # Rolling forecast para os últimos 48 pontos
    prediction_length = 24
    train_data = feature_data[:-prediction_length]
    test_data = feature_data[-prediction_length:]
    
    print(f"\\nConfiguração:")
    print(f"- Lookback: {lookback}")
    print(f"- Train data: {train_data.shape}")
    print(f"- Test data: {test_data.shape}")
    
    # Treinar
    forecaster.fit(train_data)
    
    # Rolling forecast
    rolling_predictions = []
    current_data = train_data.copy()
    
    for i in range(prediction_length):
        # Prever próximo ponto
        pred = forecaster.predict(current_data[-lookback:].reshape(1, -1))
        rolling_predictions.append(pred[0])
        
        # Atualizar dados (rolling)
        if i < prediction_length - 1:
            # Deslocar e adicionar nova previsão
            new_sample = test_data[i].copy()
            new_sample[0] = pred[0]  # Substituir target pela previsão
            current_data = np.vstack([current_data[1:], new_sample])
    
    rolling_predictions = np.array(rolling_predictions)
    actual_values = test_data[:, 0]  # Primeira coluna é o target
    
    print(f"\\nRolling forecast results:")
    print(f"- Previsões: {rolling_predictions.shape}")
    print(f"- Valores reais: {actual_values.shape}")
    print(f"- MAE: {np.mean(np.abs(rolling_predictions - actual_values)):.2f}")
    print(f"- RMSE: {np.sqrt(np.mean((rolling_predictions - actual_values)**2)):.2f}")
    
    return rolling_predictions, actual_values

if __name__ == "__main__":
    print("=== TabPFN-TS Time Series Forecasting ===\\n")
    
    # Exemplo 1: Forecasting básico
    pred1, actual1 = example_basic_forecasting()
    
    # Exemplo 2: Multivariado
    results2 = example_multivariate_forecasting()
    
    # Exemplo 3: Rolling forecast
    pred3, actual3 = example_rolling_forecast()
    
    print("\\n=== Resumo dos Resultados ===")
    print("Exemplo 1 (Básico): MAE e RMSE calculados")
    print("Exemplo 2 (Multivariado): Processadas múltiplas lojas")
    print("Exemplo 3 (Rolling): Forecast temporal sequencial")
    print("\\nTodos os exemplos concluídos!")