# Foundation Models para Time Series Forecasting com Covariáveis Dinâmicas

Este repositório contém exemplos práticos de como usar 9 modelos de foundation para forecasting de séries temporais com suporte a covariáveis dinâmicas.

## Modelos Incluídos

1. **Amazon Chronos-2** (`chronos_2_example.py`)
2. **Salesforce Moirai-2.0** (`moirai_2_example.py`)
3. **TabPFN-TS** (`tabpfn_ts_example.py`)
4. **Google TimesFM-2.5** (`timesfm_2_5_example.py`)
5. **Datadog Toto** (`toto_example.py`)
6. **IBM Flowstate** (`flowstate_example.py`)
7. **IBM TTM (TinyTimeMixers)** (`ttm_example.py`)
8. **IBM PatchTST** (`patchtst_example.py`)
9. **IBM PatchTSMixer** (`patchtsmixer_example.py`)

## Comparação dos Modelos

Execute o arquivo de comparação para ver uma análise detalhada:
```bash
python models_comparison.py
```

## Requisitos de Instalação

### Modelos Amazon
```bash
pip install chronos-forecasting
```

### Modelos Salesforce
```bash
pip install git+https://github.com/SalesforceAIResearch/uni2ts.git
```

### TabPFN-TS
```bash
pip install tabpfn-time-series
```

### Google TimesFM
```bash
git clone https://github.com/google-research/timesfm.git
cd timesfm
pip install -e .

# Para suporte a covariáveis:
uv pip install -e .[xreg]
```

### Datadog Toto
```bash
pip install toto-ts
# Opcional para performance:
pip install xFormers flash-attention
```

### Modelos IBM
```bash
pip install git+https://github.com/ibm-granite/granite-tsfm.git
```

### Modelos Transformers (PatchTST, PatchTSMixer)
```bash
pip install transformers
```

## Estrutura dos Exemplos

Cada exemplo segue um padrão similar:

1. **Configuração e Importações**: Importa as bibliotecas necessárias
2. **Classe Forecaster**: Wrapper para facilitar o uso
3. **Preparação de Dados**: Métodos para preparar dados com covariáveis
4. **Forecast**: Funções para gerar previsões
5. **Exemplos Práticos**: Demonstrações com dados sintéticos

## Tipos de Covariáveis Suportadas

### Covariáveis Passadas (Historical)
- Temperatura histórica
- Preços passados
- Métricas de sistema anteriores

### Covariáveis Futuras Conhecidas (Future Known)
- Preços programados
- Calendário de eventos
- Previsões de demanda

### Covariáveis Categóricas Estáticas
- Tipo de loja
- Região geográfica
- Categoria de produto

## Exemplos de Uso

### Amazon Chronos-2 com Covariáveis Futuras
```python
from chronos import Chronos2Pipeline

# Inicializar pipeline
pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2")

# Dados com covariáveis
context_df = pd.DataFrame({
    'id': ['store_1'] * 100,
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
    'target': sales_data,
    'price': price_data,  # Covariável passada
})

future_df = pd.DataFrame({
    'id': ['store_1'] * 24,
    'timestamp': pd.date_range('2024-01-05', periods=24, freq='H'),
    'price': future_prices,  # Covariável futura conhecida
})

# Forecast
predictions = pipeline.predict_df(
    context_df=context_df,
    future_df=future_df,
    prediction_length=24,
    id_column='id',
    timestamp_column='timestamp',
    target='target'
)
```

### IBM TTM com Variáveis Exógenas
```python
from tsfm_public import TinyTimeMixerForPrediction
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor

# Pré-processar dados
preprocessor = TimeSeriesPreprocessor(
    freq='D',
    target_cols=['sales'],
    past_exogenous_cols=['temperature', 'promotion'],
    future_exogenous_cols=['price'],
    static_categorical_cols=['store_type']
)

preprocessed_data = preprocessor.fit_transform(data)

# Forecast
model = TinyTimeMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-ttm-r2")
# ... (configuração e forecast)
```

### Salesforce Moirai-2.0
```python
from uni2ts.model.moirai import Moirai2Forecast

# Configurar modelo
model = Moirai2Forecast(
    patch_size=64,
    context_length=512,
    prediction_length=96,
    feat_dynamic_real_dim=2,  # Número de covariáveis futuras
    past_feat_dynamic_real_dim=3,  # Número de covariáveis passadas
)

# Dataset com covariáveis
dataset = PandasDataset(
    dataframe=data,
    target=['sales'],
    timestamp='date',
    feat_dynamic_real=['price'],  # Futuras
    past_feat_dynamic_real=['temperature', 'promotion']  # Passadas
)
```

## Considerações Importantes

### 1. Escalonamento de Dados
- Muchos modelos requerem normalização dos dados
- Use sempre o mesmo scaler para treino e teste
- IBM TTM inclui utilitários para pré-processamento

### 2. Contexto vs Horizonte
- **Contexto**: Quantos pontos históricos usar
- **Horizonte**: Quantos pontos futuros prever
- Configurações diferentes afetam performance

### 3. Covariáveis Dinâmicas
- **Futuras**: Devem ser conhecidas no momento da previsão
- **Passadas**: Podem incluir até o último ponto do contexto
- **Categóricas**: Úteis para diferenciação entre séries

### 4. Recursos Computacionais
- Modelos grandes: Google TimesFM, Chronos-2 (GPU recomendada)
- Modelos compactos: IBM TTM, PatchTST (CPU possível)
- TabPFN-TS: Requer conexão com servidor

## Troubleshooting

### Erro de Memória
- Reduza o context_length
- Use batch_size menor
- Processe uma série por vez

### Erro de Instalação
```bash
# Para modelos IBM
pip install --upgrade pip setuptools wheel

# Para TimesFM
conda create -n timesfm python=3.9
conda activate timesfm
pip install torch torchvision torchaudio
pip install -e .

# Para Chronos
pip install --upgrade torch
```

### Performance Lenta
- Use GPU quando disponível
- Considere modelos compactos (IBM TTM, PatchTST)
- Para TabPFN-TS, use client local

## Benchmarks e Performance

### GIFT-Eval (General Time Series Forecasting)
1. **TabPFN-TS** (1º lugar)
2. **IBM Flowstate** (Melhor Zero-Shot)
3. **Amazon Chronos-2**

### BOOM (Observability)
1. **Datadog Toto** (Especializado)
2. **Amazon Chronos-2**
3. **Google TimesFM-2.5**

### Latência (menor é melhor)
1. **IBM TTM** (mais rápido)
2. **TabPFN-TS**
3. **IBM PatchTST**

## Quando Usar Cada Modelo

### Forecasting com Covariáveis Futuras
- **Amazon Chronos-2**: Melhor suporte nativo
- **Salesforce Moirai-2.0**: Excelente para casos complexos
- **IBM TTM**: Compacto e eficiente

### Observabilidade e DevOps
- **Datadog Toto**: Especializado para métricas
- **Amazon Chronos-2**: Alternativa geral
- **Google TimesFM-2.5**: Para pesquisa

### Produção com Recursos Limitados
- **IBM TTM**: < 1M parâmetros
- **IBM PatchTST**: Eficiente para long-term
- **IBM Flowstate**: Multi-escala temporal

### Zero-Shot (Sem Treinamento)
- **TabPFN-TS**: Líder em benchmarks
- **Amazon Chronos-2**: SOTA em muitos casos
- **IBM Flowstate**: Inovador

## Contribuindo

Para adicionar novos exemplos ou melhorar os existentes:

1. Fork o repositório
2. Adicione novos modelos ou melhorias
3. Teste com diferentes tipos de dados
4. Documente covariáveis e configurações
5. Submit pull request

## Licença

Os exemplos seguem as licenças dos modelos originais. Consulte os repositórios oficiais para termos específicos.

## Referências

- [Amazon Chronos Paper](https://arxiv.org/abs/2503.19789)
- [Salesforce Moirai Paper](https://arxiv.org/abs/2402.02592)
- [TabPFN-TS Paper](https://arxiv.org/abs/2501.02945)
- [Google TimesFM Paper](https://arxiv.org/abs/2310.10688)
- [Datadog Toto Paper](https://arxiv.org/abs/2505.14766)
- [IBM Flowstate Paper](https://arxiv.org/abs/2508.05287)
- [IBM TTM Paper](https://arxiv.org/abs/2401.03955)
- [IBM PatchTST Paper](https://arxiv.org/abs/2211.14730)
- [IBM PatchTSMixer Paper](https://arxiv.org/abs/2306.09364)

---

**Nota**: Este repositório é para fins educacionais e de pesquisa. Para uso em produção, consulte a documentação oficial de cada modelo e considere fatores como licenciamento, compliance e performance específicos.