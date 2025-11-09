Guia de Implementação de Covariáveis Dinâmicas para Modelos de Fundação de Séries Temporais Zero-ShotIntrodução: O Desafio das Covariáveis Dinâmicas em TSFMsModelos de Fundação de Séries Temporais (TSFMs) pré-treinados, como Chronos, Moirai e TimesFM, representam uma mudança de paradigma na previsão. Eles oferecem capacidades de zero-shot (ou few-shot), permitindo previsões razoavelmente precisas em séries temporais nunca vistas, sem a necessidade de re-treinamento específico por conjunto de dados.1No entanto, a implementação prática desses modelos encontra um obstáculo significativo quando confrontada com dados do mundo real: covariáveis dinâmicas. Estas são variáveis exógenas (externas) que mudam ao longo do tempo e influenciam a série temporal alvo (por exemplo, prever vendas de eletricidade usando a temperatura futura conhecida).A consulta do usuário visa preencher essa lacuna crítica entre a teoria e a prática. Embora a maioria dos tutoriais demonstre a previsão univariada, o uso eficaz de covariáveis dinâmicas (também conhecidas como feat_dynamic_real, known_future_covariates, ou XReg) não é padronizado.Esta análise técnica examina os nove TSFMs solicitados e revela que seus métodos de implementação para covariáveis dinâmicas se enquadram em quatro padrões arquitetônicos distintos:APIs de DataFrame Explícitas: Modelos que expõem um argumento de API de alto nível (por exemplo, future_df=...) para ingerir diretamente DataFrames pandas contendo covariáveis futuras.Abstração de Dataset (Ecossistema GluonTS): Modelos que delegam o manuseio de covariáveis a uma camada de abstração de dados, definindo papéis (alvo vs. covariável) na criação do dataset, e não na chamada de predict.Abordagem de Canal Multivariado: Modelos que não distinguem tecnicamente entre alvos e covariáveis. Ambos são tratados como "canais" em um tensor multivariado, exigindo que o usuário formate a entrada e fatie a saída manualmente.Abordagem de Tabularização: Um paradigma único que transforma o problema de sequência em um problema de regressão tabular, onde as covariáveis são simplesmente features (colunas) em uma tabela.Este relatório fornecerá um exemplo de script Python e uma análise técnica para cada modelo, agrupados por esses padrões de implementação.Padrão 1 (Parte A): Amazon Chronos-2 — A API future_dfO Amazon Chronos-2 (amazon/chronos-2) é um TSFM de 120M de parâmetros que, em sua segunda versão, introduziu suporte nativo de primeira classe para previsão informada por covariáveis.4 Versões anteriores do Chronos não suportavam nativamente covariáveis, muitas vezes exigindo modelos de regressão externos.6 O Chronos-2 resolve isso diretamente.A abordagem do Chronos-2 se enquadra no padrão de "API Explícita", caracterizado por sua simplicidade e uso direto do pandas.Ambiente e InstalaçãoA funcionalidade de covariável requer a versão 2.0 ou superior da biblioteca chronos-forecasting.Python# É crucial instalar a versão >= 2.0 para suporte a covariáveis
!pip install "chronos-forecasting>=2.0" pandas pyarrow
Análise do Método de ImplementaçãoA simplicidade do Chronos-2 reside na sua API predict_df de alto nível.4 Esta função aceita DataFrames pandas brutos e abstrai internamente todo o alinhamento, normalização e tokenização necessários.O padrão requer dois DataFrames principais:context_df: Um DataFrame contendo o histórico de tempo, a(s) coluna(s) alvo (target) e o histórico de todas as colunas covariadas.future_df: Um DataFrame opcional contendo os valores futuros conhecidos das covariáveis. Este DataFrame não deve incluir a coluna alvo.4O pipeline alinha inteligentemente os timestamps e usa os nomes das colunas (especificados nos argumentos) para discernir entre alvos e covariáveis.Script de Exemplo: Chronos-2 com Covariáveis DinâmicasEste script demonstra a previsão do preço da eletricidade (target) usando covariáveis futuras conhecidas (por exemplo, demanda de eletricidade prevista, dados meteorológicos).Pythonimport pandas as pd
import numpy as np
import torch
from chronos import Chronos2Pipeline

# 1. Carregar o Pipeline
# Use "cpu" se a GPU não estiver disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map=device,
    torch_dtype=torch.bfloat16,
)

# 2. Preparar Dados de Exemplo
# Criamos um DataFrame de exemplo no formato "longo"
PREDICTION_LENGTH = 24
CONTEXT_LENGTH = 96
TIMESTAMPS = pd.date_range(start="2023-01-01", periods=CONTEXT_LENGTH + PREDICTION_LENGTH, freq="H")

# Simular dados para dois itens (duas séries temporais independentes)
item_1_data = {
    "id": "item_1",
    "timestamp": TIMESTAMPS,
    "target": np.sin(np.arange(0, CONTEXT_LENGTH + PREDICTION_LENGTH) * 0.1) + np.random.rand(CONTEXT_LENGTH + PREDICTION_LENGTH) * 0.2,
    "known_covariate": np.cos(np.arange(0, CONTEXT_LENGTH + PREDICTION_LENGTH) * 0.1), # Covariável futura conhecida
    "past_covariate": np.random.rand(CONTEXT_LENGTH + PREDICTION_LENGTH) # Covariável apenas passada
}
item_2_data = {
    "id": "item_2",
    "timestamp": TIMESTAMPS,
    "target": np.sin(np.arange(0, CONTEXT_LENGTH + PREDICTION_LENGTH) * 0.2) + np.random.rand(CONTEXT_LENGTH + PREDICTION_LENGTH) * 0.3,
    "known_covariate": np.cos(np.arange(0, CONTEXT_LENGTH + PREDICTION_LENGTH) * 0.2),
    "past_covariate": np.random.rand(CONTEXT_LENGTH + PREDICTION_LENGTH)
}

full_df = pd.concat().reset_index()

# 3. Criar os DataFrames context_df e future_df
# context_df contém o histórico (alvo + covariáveis)
context_df = full_df.iloc # (CONTEXT_LENGTH * 2 itens)

# future_df contém apenas timestamps futuros e covariáveis futuras conhecidas
# É crucial remover a coluna 'target'
future_df = full_df.iloc[["id", "timestamp", "known_covariate"]]

print(f"Context DF shape: {context_df.shape}")
print(f"Future DF shape: {future_df.shape}")

# 4. Executar Inferência com Covariáveis
# O pipeline lida com o formato longo (id_column) automaticamente
pred_df = pipeline.predict_df(
    context_df,
    future_df=future_df,
    prediction_length=PREDICTION_LENGTH,
    id_column="id",
    timestamp_column="timestamp",
    target="target",
    # known_covariates e past_covariates são inferidos das colunas
    # que não são 'id', 'timestamp' ou 'target'
)

# 5. Analisar Saída
# pred_df conterá as previsões (quantis) para 'item_1' e 'item_2'
print("\n--- Previsão (pred_df) ---")
print(pred_df.head())
print(f"\nShape da previsão: {pred_df.shape}")
Padrão 1 (Parte B): Google TimesFM 2.5 — A Abordagem "XReg" HíbridaO Google TimesFM (google/timesfm-2.5-200m-pytorch) é um modelo decoder-only.8 Após uma atualização (Outubro de 2025), o suporte a covariáveis foi "adicionado de volta" usando uma abordagem de regressores externos (XReg).10Isso o coloca na categoria "API Explícita", mas com uma implementação fundamentalmente diferente do Chronos-2.Ambiente e InstalaçãoO suporte a covariáveis é um módulo opcional que deve ser instalado explicitamente. A funcionalidade XReg depende de jax e jaxlib, mesmo ao usar a versão PyTorch do TimesFM.11Python# 1. Clonar o repositório oficial
!git clone https://github.com/google-research/timesfm.git
%cd timesfm

# 2. Instalar com o extra [xreg]
# Isso instala dependências jax/flax necessárias para covariáveis
!pip install -e ".[xreg, torch]"
Análise do Método de ImplementaçãoO TimesFM não é um modelo de ponta a ponta para covariáveis. Em vez disso, ele usa uma arquitetura híbrida.12 A função model.forecast_with_covariates 11:Usa um modelo de regressão externo (XReg) para ajustar e prever as covariáveis.Calcula os resíduos ( target - XReg_prediction ).Usa o modelo de fundação TimesFM para prever os resíduos futuros.A previsão final é a soma: XReg_future_prediction + TimesFM_residual_forecast.Devido a essa abordagem, o usuário deve classificar explicitamente as covariáveis em dicionários: dynamic_numerical_covariates e dynamic_categorical_covariates.11 Crucialmente, os dados da covariável devem abranger tanto o contexto quanto o horizonte.11Script de Exemplo: TimesFM 2.5 com Covariáveis DinâmicasEste script usa uma API de nível inferior (arrays numpy) e demonstra a separação explícita de tipos de covariáveis.Pythonimport numpy as np
import torch
import timesfm

# 1. Configurar o Modelo
# Instalar com ".[xreg]" é um pré-requisito
model = timesfm.TimesFM_2.5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

# Compilar o modelo [9]
model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
    )
)

# 2. Preparar Dados de Exemplo
PREDICTION_LENGTH = 30
CONTEXT_LENGTH = 120
FULL_LENGTH = CONTEXT_LENGTH + PREDICTION_LENGTH

# Criar dados de entrada
# O TimesFM espera uma lista de arrays 1D
past_target = np.sin(np.arange(0, CONTEXT_LENGTH) * 0.1)

# Covariáveis DEVEM cobrir o comprimento TOTAL (contexto + horizonte)
# Covariável Numérica Dinâmica (ex: temperatura)
future_temp = np.cos(np.arange(0, FULL_LENGTH) * 0.1)
# Covariável Categórica Dinâmica (ex: dia da semana)
future_weekday = np.arange(0, FULL_LENGTH) % 7

print(f"Shape do Alvo Passado: {past_target.shape}")
print(f"Shape da Covariável Numérica: {future_temp.shape}")
print(f"Shape da Covariável Categórica: {future_weekday.shape}")

# 3. Executar Inferência com Covariáveis
# Usamos 'forecast_with_covariates'
# As entradas são listas de arrays (para processamento em lote)
point_forecast, quantile_forecast = model.forecast_with_covariates(
    inputs=[past_target],
    horizon=PREDICTION_LENGTH,
    
    # Dicionário de covariáveis numéricas
    dynamic_numerical_covariates={
        "temperature": [future_temp] # Deve ser uma lista de arrays
    },
    
    # Dicionário de covariáveis categóricas
    dynamic_categorical_covariates={
        "weekday": [future_weekday] # Deve ser uma lista de arrays
    },

    # Também suporta covariáveis estáticas (ex: ID da loja)
    static_categorical_covariates={}, 
)

# 4. Analisar Saída
# A saída é uma lista de arrays
forecast_array = point_forecast
print(f"\nShape da Previsão: {forecast_array.shape}") # (30,)
Padrão 2: Salesforce Moirai — A Abordagem feat_dynamic_realO Moirai da Salesforce (por exemplo, Salesforce/moirai-2.0-R-small) faz parte da biblioteca uni2ts e é projetado para previsão "any-variate" (qualquer-variável).1Sua abordagem para covariáveis é um exemplo do padrão de "Abstração de Dataset". A complexidade é movida da chamada de predict para a etapa de ingestão de dados, que depende fortemente do ecossistema GluonTS.Ambiente e InstalaçãoO Moirai é usado através da biblioteca uni2ts, que deve ser clonada e instalada a partir da fonte.Python# 1. Clonar o repositório uni2ts
!git clone https://github.com/SalesforceAIResearch/uni2ts.git
%cd uni2ts

# 2. Instalar com dependências de notebook (inclui gluonts)
!pip install -e ".[notebook]"
!touch.env # Necessário pelo uni2ts
Análise do Método de ImplementaçãoO Moirai não aceita um future_df em sua função predict. Em vez disso, o usuário deve formatar os dados em um PandasDataset do GluonTS.15 Durante a criação deste dataset, o usuário especifica quais colunas são o target e quais são feat_dynamic_real (covariáveis dinâmicas reais).16O objeto PandasDataset então gerencia internamente o fatiamento e a alimentação dos dados corretos (histórico do alvo, histórico da covariável e futuro da covariável) para o modelo durante a inferência. O modelo MoiraiForecast é inicializado com as dimensões dessas covariáveis (por exemplo, feat_dynamic_real_dim=1), informando-o para esperar esses dados do dataset.13Script de Exemplo: Moirai com Covariáveis DinâmicasEste script demonstra o fluxo de trabalho gluonts necessário, usando um DataFrame em "formato longo".Pythonimport pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt

# 1. Preparar Dados em Formato Longo (como o Chronos)
PREDICTION_LENGTH = 24
CONTEXT_LENGTH = 96
TIMESTAMPS = pd.date_range(start="2023-01-01", periods=CONTEXT_LENGTH + PREDICTION_LENGTH, freq="H")

data = {
    "item_id": "A",
    "timestamp": TIMESTAMPS,
    "target_sales": np.sin(np.arange(0, CONTEXT_LENGTH + PREDICTION_LENGTH) * 0.1) + np.random.rand(CONTEXT_LENGTH + PREDICTION_LENGTH) * 0.2,
    "future_promo": np.cos(np.arange(0, CONTEXT_LENGTH + PREDICTION_LENGTH) * 0.1), # Covariável futura conhecida
}
df = pd.DataFrame(data)

# 2. A Etapa Crítica: Criar o PandasDataset
# Definimos as covariáveis AQUI, não na chamada predict()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp') # GluonTS requer um índice de data e hora

dataset = PandasDataset.from_long_dataframe(
    df,
    target='target_sales',
    item_id='item_id',
    feat_dynamic_real=['future_promo'] # <-- Definição da covariável
)

# 3. Dividir os dados usando a utilidade do GluonTS
# O Moirai fará a previsão zero-shot nos dados de 'treinamento'
train_data, test_gen = split(dataset, offset=-PREDICTION_LENGTH)

# 4. Carregar o Modelo Moirai
# O modelo Moirai 2.0 (decoder-only)
model_id = "Salesforce/moirai-2.0-R-small"
model = MoiraiForecast(
    module=MoiraiModule.from_pretrained(model_id),
    prediction_length=PREDICTION_LENGTH,
    context_length=CONTEXT_LENGTH,
    
    # O modelo deve ser informado sobre as dimensões
    target_dim=1,
    feat_dynamic_real_dim=dataset.num_feat_dynamic_real, # Obtido do dataset
    past_feat_dynamic_real_dim=0
)

# 5. Executar Inferência
# O predictor.predict() puxa implicitamente as covariáveis do 'train_data'
predictor = model.create_predictor(batch_size=32)
forecasts = list(predictor.predict(train_data))

# 6. Analisar Saída
print(f"Número de previsões: {len(forecasts)}")
# O objeto de previsão contém os quantis
print(f"Previsão média para o primeiro passo: {forecasts.mean}")
# forecasts.plot() # Requer matplotlib
Padrão 3: A Abordagem do Canal Multivariado (Datadog Toto e IBM)Este padrão é o mais fundamental e comum entre os TSFMs de nível inferior. Ele trata o problema de "previsão com covariáveis" como um simples problema de "previsão multivariada".Não há distinção entre um "alvo" e uma "covariável" na entrada do modelo. Ambos são simplesmente "canais".O fluxo de trabalho é:Entrada: Formatar os dados como um tensor ``.canal_0 = targetcanal_1 = covariate_Acanal_2 = covariate_BInferência: Alimentar este tensor no modelo.Saída: O modelo retorna um tensor de previsão de forma ``. Ele prevê todos os canais.Pós-processamento: O usuário deve manualmente fatiar a saída para extrair apenas a previsão do canal alvo (ou seja, output_tensor[0, :]).Subseção 5.1: Datadog Toto (Datadog/Toto-Open-Base-1.0)O Toto é um TSFM de 151M de parâmetros otimizado para dados de observabilidade, que são inerentemente multivariados.18 Sua API toto-ts espera tensores torch puros.Ambiente e InstalaçãoPython!pip install toto-ts
# Para velocidade ideal, também instale xformers e flash-attention
#!pip install xformers flash-attention
Script de Exemplo: Toto com Múltiplos CanaisEste script usa a API MaskedTimeseries de nível de tensor do Toto.Pythonimport torch
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto

# 1. Configurar Modelo e Dispositivo
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0').to(DEVICE)
toto.compile() # Opcional, para velocidade
forecaster = TotoForecaster(toto.model)

# 2. Preparar Dados de Entrada (Formato de Canal)
CONTEXT_LEN = 512
PRED_LEN = 96
N_CHANNELS = 2 # 1 (alvo) + 1 (covariável)

# Criar um tensor
# Canal 0: Alvo (ex: uso da CPU)
# Canal 1: Covariável (ex: nº de requisições)
past_data = torch.randn(N_CHANNELS, CONTEXT_LEN).to(DEVICE)

# O Toto espera um objeto MaskedTimeseries
# Outros campos são principalmente para preenchimento/mascaramento
inputs = MaskedTimeseries(
    series=past_data,
    padding_mask=torch.full_like(past_data, True, dtype=torch.bool),
    id_mask=torch.zeros_like(past_data), # Não usado na inferência básica
    timestamp_seconds=torch.zeros_like(past_data), # Não usado pelo modelo base
    time_interval_seconds=torch.full((N_CHANNELS,), 60).to(DEVICE) # metadata
)

# 3. Executar Inferência
# O modelo irá prever TODOS os N_CANAIS
forecast = forecaster.forecast(
    inputs,
    prediction_length=PRED_LEN,
    num_samples=256,
)

# 4. Analisar Saída (A Etapa Crítica)
# 'forecast.median' tem a forma
full_prediction_tensor = forecast.median
print(f"Shape da previsão completa: {full_prediction_tensor.shape}") # (2, 96)

# Extrair APENAS a previsão do canal alvo (índice 0)
target_forecast = full_prediction_tensor[0, :]

# A previsão da covariável (índice 1) é descartada
covariate_forecast_discarded = full_prediction_tensor[1, :]

print(f"Shape da previsão do alvo extraído: {target_forecast.shape}") # (96,)
Subseção 5.2: A Suíte IBM granite-tsfmOs modelos da IBM (Flowstate, TTM, PatchTST, PatchTSMixer) são todos liberados sob a biblioteca unificada granite-tsfm.21 Esta biblioteca usa o mesmo padrão de canal multivariado do Toto, mas fornece uma camada de pré-processamento de alto nível (TimeSeriesPreprocessor) para converter DataFrames pandas no formato de tensor de canal necessário.Ambiente e Instalação Unificada (IBM)Python# Instala a biblioteca 'granite-tsfm' e dependências de notebook
!pip install "granite-tsfm[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git"
!pip install transformers accelerate # Necessário para PatchTST/PatchTSMixer
Preparação de Dados Unificada (IBM)Este é o fluxo de trabalho pandas-para-tensor usado por todos os quatro modelos da IBM.Pythonimport pandas as pd
import numpy as np
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.dataset import ForecastDFDataset
import torch

# 1. Criar DataFrame (Alvo + Covariável como colunas)
CONTEXT_LEN = 512
PRED_LEN = 96
N_CHANNELS = 2

data_df = pd.DataFrame({
    "timestamp": pd.date_range(start="2020-01-01", periods=CONTEXT_LEN, freq="H"),
    "target_load": np.random.rand(CONTEXT_LEN),
    "covariate_temp": np.random.rand(CONTEXT_LEN)
})

# 2. Configurar o Preprocessor
# A Etapa Crítica: Tratar TODAS as colunas como 'input_columns' E 'output_columns'
# Isso diz ao preprocessor para criar um problema MIMO (Multiple-Input, Multiple-Output)
forecast_columns = ["target_load", "covariate_temp"]

preprocessor = TimeSeriesPreprocessor(
    timestamp_column="timestamp",
    id_columns=, # Sem múltiplos IDs de item neste exemplo
    input_columns=forecast_columns, # <-- Usar ambos como entrada
    output_columns=forecast_columns, # <-- Prever ambos
    scaling=True
)

# "Treinar" o preprocessor (aprende estatísticas de escalonamento)
preprocessor = preprocessor.train(data_df)

# 3. Criar o PyTorch Dataset
# O dataset irá gerar lotes com N_CHANNELS=2
dataset = ForecastDFDataset(
    preprocessor.preprocess(data_df),
    id_columns=,
    timestamp_column="timestamp",
    input_columns=forecast_columns,
    output_columns=forecast_columns,
    context_length=CONTEXT_LEN,
    prediction_length=PRED_LEN,
)

# Obter um único lote (batch) de exemplo
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
batch = next(iter(dataloader))

# 'past_values' é o tensor de entrada para os modelos
# A forma é ou
print(f"Forma do lote de entrada (past_values): {batch['past_values'].shape}")
Script de Exemplo: IBM Flowstate (ibm-research/flowstate)O Flowstate é um modelo de Espaço de Estado (SSM).23 Ele espera a forma de entrada ``.24Pythonfrom tsfm_public.models.flowstate import FlowStateForPrediction

# 1. Carregar Modelo
model = FlowStateForPrediction.from_pretrained("ibm-research/flowstate")
model.eval()

# 2. Obter Lote (do passo anterior de preparação de dados)
# O dataloader 'dataset' já está configurado para 2 canais
batch = next(iter(dataloader))
# O Flowstate espera (Context, Batch, Channels)
past_values = batch['past_values'].permute(1, 0, 2) # Transpor B, L, C -> L, B, C
print(f"Forma de entrada do Flowstate: {past_values.shape}")

# 3. Inferência
with torch.no_grad():
    outputs = model(
        past_values,
        prediction_length=PRED_LEN,
        batch_first=False # Indica que o Lote não é a primeira dimensão
    )

# 4. Fatiar Saída
# Saída:
print(f"Forma da saída completa: {outputs.prediction_outputs.shape}")
# Extrair previsão do canal 0 (target_load), mediana (índice 4 se 9 quantis)
target_forecast = outputs.prediction_outputs[:, 4, :, 0] 
print(f"Forma da previsão do alvo: {target_forecast.shape}")
Script de Exemplo: IBM TinyTimeMixer (TTM) (ibm/TTM)O TTM é um modelo leve baseado em MLP.25 Ele espera a forma ``.Pythonfrom tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

# 1. Carregar Modelo
# Nota: TTMs são pré-treinados para pares específicos (Contexto, Horizonte)
# Usaremos um modelo genérico (não listado na consulta) para fins de API
# ou um modelo ajustado, mas a API é a mesma.
# Para este exemplo, vamos carregar a configuração base
model = TinyTimeMixerForPrediction.from_pretrained("ibm-granite/granite-timeseries-ttm-r2")
model.eval()

# 2. Obter Lote (do passo anterior de preparação de dados)
batch = next(iter(dataloader))
past_values = batch['past_values'] # Forma
print(f"Forma de entrada do TTM: {past_values.shape}")

# 3. Inferência
with torch.no_grad():
    outputs = model(
        past_values=past_values,
        future_values=batch['future_values'] # Usado para ground truth, não inferência
    )

# 4. Fatiar Saída
# Saída:
print(f"Forma da saída completa: {outputs.prediction_outputs.shape}")
# Extrair previsão do canal 0 (target_load)
target_forecast = outputs.prediction_outputs[:, :, 0]
print(f"Forma da previsão do alvo: {target_forecast.shape}")
Script de Exemplo: IBM PatchTST (ibm/patchtst-base-prediction-etth1)O PatchTST é um modelo baseado em Transformer.27 A biblioteca transformers da Hugging Face tem integração nativa com o granite-tsfm.29 A forma de entrada é ``.Pythonfrom transformers import PatchTSTForPrediction

# 1. Carregar Modelo
# O modelo 'etth1' foi pré-treinado em 7 canais
# Podemos fazer fine-tuning ou zero-shot, mas a entrada multivariada é a mesma
model = PatchTSTForPrediction.from_pretrained("ibm/patchtst-base-prediction-etth1")
model.eval()

# 2. Obter Lote (do passo anterior de preparação de dados)
batch = next(iter(dataloader))
past_values = batch['past_values'] # Forma
print(f"Forma de entrada do PatchTST: {past_values.shape}")

# 3. Inferência
with torch.no_grad():
    outputs = model(
        past_values=past_values,
    )

# 4. Fatiar Saída
# Saída:
print(f"Forma da saída completa: {outputs.prediction_outputs.shape}")
# Extrair previsão do canal 0 (target_load)
target_forecast = outputs.prediction_outputs[:, :, 0]
print(f"Forma da previsão do alvo: {target_forecast.shape}")
Script de Exemplo: IBM PatchTSMixer (ibm/patchtsmixer-base-prediction-etth1)O PatchTSMixer é um modelo leve baseado em MLP-Mixer.30 Sua API é idêntica à do PatchTST. A forma de entrada é ``.Pythonfrom transformers import PatchTSMixerForPrediction

# 1. Carregar Modelo
model = PatchTSMixerForPrediction.from_pretrained("ibm/patchtsmixer-base-prediction-etth1")
model.eval()

# 2. Obter Lote (do passo anterior de preparação de dados)
batch = next(iter(dataloader))
past_values = batch['past_values'] # Forma
print(f"Forma de entrada do PatchTSMixer: {past_values.shape}")

# 3. Inferência
with torch.no_grad():
    outputs = model(
        past_values=past_values,
    )

# 4. Fatiar Saída
# Saída:
print(f"Forma da saída completa: {outputs.prediction_outputs.shape}")
# Extrair previsão do canal 0 (target_load)
target_forecast = outputs.prediction_outputs[:, :, 0]
print(f"Forma da previsão do alvo: {target_forecast.shape}")
Padrão 4: PriorLabs TabPFN-TS — A Abordagem de TabularizaçãoO TabPFN-TS (PriorLabs/TabPFN-TS) é o modelo mais singular desta lista. Ele não é um modelo de sequência. Em vez disso, ele reformula a previsão de séries temporais como um problema de regressão tabular e aplica o TabPFN (um modelo de fundação para dados tabulares) para resolvê-lo.32Ambiente e InstalaçãoPython!pip install tabpfn-time-series
Análise do Método de ImplementaçãoO modelo adere a uma API sklearn-like: model.fit(X_train, y_train) e model.predict(X_test).35Neste paradigma:Não há noção de sequência: O modelo depende inteiramente da engenharia de features (por exemplo, dia_da_semana, mês_do_ano) para capturar a temporalidade.36Covariáveis são Apenas Features: Uma "covariável dinâmica" é simplesmente outra coluna (feature) na tabela de entrada.2X_train e y_train são o contexto (passado). X_train contém features de tempo passado e covariáveis passadas; y_train contém o alvo passado.X_test são as features futuras. Ele deve conter as features de tempo futuro e as covariáveis futuras conhecidas.O fit() realiza a inferência "in-context" no histórico, e o predict() aplica o padrão aprendido às features futuras.Script de Exemplo: TabPFN-TS com Covariáveis como FeaturesEste script mostra o fluxo de trabalho sklearn-like.Pythonimport pandas as pd
import numpy as np
from tabpfn_time_series import TabPFNTimeSeriesRegressor

# 1. Preparar Dados Tabulares (Passado e Futuro)
CONTEXT_LEN = 100
PRED_LEN = 20
FULL_LEN = CONTEXT_LEN + PRED_LEN

# Criar features de tempo e covariáveis
time_idx = np.arange(FULL_LEN)
# Feature de tempo (ex: dia da semana)
time_feat_weekday = time_idx % 7 
# Covariável futura conhecida (ex: promoção)
future_cov_promo = (time_idx % 30 < 5).astype(int) 
# Alvo (depende da covariável e do tempo)
target = (time_feat_weekday * 0.1) + (future_cov_promo * 0.5) + np.random.rand(FULL_LEN) * 0.1

# Criar DataFrame tabular
df = pd.DataFrame({
    'time_idx': time_idx,
    'weekday': time_feat_weekday,
    'promo': future_cov_promo, # Esta é a nossa covariável
    'target': target
})

# 2. Dividir em "fit" (passado) e "predict" (futuro)
df_fit = df.iloc
df_predict = df.iloc

# 3. Separar X e y para o conjunto de "fit"
X_fit = df_fit.drop(columns=['target'])
y_fit = df_fit['target']

# X para o conjunto de "predict" (NÃO inclui o alvo)
X_predict = df_predict.drop(columns=['target'])

print(f"Forma de X_fit: {X_fit.shape}") # (100, 3)
print(f"Forma de y_fit: {y_fit.shape}") # (100,)
print(f"Forma de X_predict: {X_predict.shape}") # (20, 3)

# 4. Inicializar Modelo e Executar "Fit/Predict"
# N_ensemble_configurations define o trade-off velocidade/precisão
model = TabPFNTimeSeriesRegressor(N_ensemble_configurations=4)

# "Treinar" (fit) no contexto histórico (executa inferência in-context)
model.fit(X_fit, y_fit)

# Prever usando as features futuras (incluindo a covariável futura 'promo')
prediction = model.predict(X_predict)

# 5. Analisar Saída
print(f"\nForma da Previsão: {prediction.shape}") # (20,)
print("Previsão:")
print(prediction)
Análise Comparativa e Recomendações FinaisA análise dos nove modelos revela uma falta crítica de padronização na indústria de TSFM para o manuseio de covariáveis. A escolha de um modelo impacta diretamente o pipeline de engenharia de dados.A tabela a seguir resume os quatro paradigmas identificados e os métodos de API específicos para cada modelo.Tabela 1: Resumo dos Métodos de Ingestão de Covariáveis DinâmicasModeloBiblioteca/Pacote ChaveMétodo/Função de IngestãoParadigma de Dados de EntradaAmazon Chronos-2chronos-forecastingpipeline.predict_df()context_df (DataFrame) e future_df (DataFrame) 4Google TimesFM 2.5timesfmmodel.forecast_with_covariates()Dicionários (ex: dynamic_numerical_covariates) 11Salesforce Moiraiuni2ts (via gluonts)PandasDataset.from_long_dataframe()Argumento feat_dynamic_real no construtor do dataset 16Datadog Totototo-tsforecaster.forecast()MaskedTimeseries (Tensor torch [n_channels, L]) 19IBM Flowstategranite-tsfmmodel()Tensor (via TimeSeriesPreprocessor) `` 24IBM TinyTimeMixergranite-tsfmmodel()Tensor (via TimeSeriesPreprocessor) ``IBM PatchTSTgranite-tsfmmodel()Tensor (via TimeSeriesPreprocessor) `` 29IBM PatchTSMixergranite-tsfmmodel()Tensor (via TimeSeriesPreprocessor) `` 30PriorLabs TabPFN-TStabpfn-time-seriesmodel.fit() / model.predict()DataFrame Tabular (Covariáveis = Colunas) 2Recomendações FinaisPara Facilidade de Uso e Integração com pandas: Amazon Chronos-2 oferece a API mais limpa e de mais alto nível. Sua abordagem future_df é intuitiva e ideal para rápida prototipagem.Para Ecossistemas GluonTS Existentes: Salesforce Moirai é a escolha natural. Ele se integra perfeitamente ao fluxo de trabalho do PandasDataset, embora exija a adoção desse paradigma de preparação de dados.Para Controle Híbrido (Regressão + Resíduos): Google TimesFM 2.5 é único. Sua abordagem XReg permite que especialistas modelem explicitamente o impacto da covariável com um regressor, deixando o TSFM focar nos padrões residuais mais complexos.Para Controle de Nível de Tensor e Flexibilidade Multivariada: Datadog Toto (para torch puro) e a suíte IBM granite-tsfm (para um wrapper pandas-para-tensor robusto) são as mais poderosas. Elas tratam covariáveis como canais, oferecendo flexibilidade máxima, mas exigem que o usuário gerencie manualmente o fatiamento da saída.Para Velocidade de Inferência Rápida e Problemas Tabulares: TabPFN-TS é a escolha mais rápida. É ideal para cenários onde a engenharia de features (derivação de dia_da_semana, promoção, etc.) é mais importante do que a modelagem de sequência complexa.