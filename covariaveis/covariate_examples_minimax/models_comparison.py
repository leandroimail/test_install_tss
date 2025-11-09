"""
Comparação dos 9 Modelos de Foundation para Time Series Forecasting

Este documento apresenta uma visão comparativa dos modelos pesquisados,
com foco em suas capacidades de forecasting com covariáveis dinâmicas.
"""

import pandas as pd
import numpy as np

# Dados de comparação dos modelos
models_comparison = {
    'Modelo': [
        'Amazon Chronos-2',
        'Salesforce Moirai-2.0-R-small',
        'TabPFN-TS',
        'Google TimesFM-2.5',
        'Datadog Toto-Open-Base-1.0',
        'IBM Flowstate',
        'IBM TTM (TinyTimeMixers)',
        'IBM PatchTST',
        'IBM PatchTSMixer'
    ],
    
    'Parâmetros': [
        '120M',
        '14M-311M',
        '~100M (TabPFNv2)',
        '200M',
        '151M',
        '<10M',
        '<1M',
        '616k',
        '196k'
    ],
    
    'Arquitetura': [
        'Encoder-only Transformer',
        'Decoder-only Transformer',
        'Tabular Regression (TabPFNv2)',
        'Decoder-only Transformer',
        'Decoder-only Transformer',
        'SSM Encoder + Functional Basis Decoder',
        'MLP-based',
        'Transformer with Patching',
        'MLP-Mixer'
    ],
    
    'Covariáveis_Dinâmicas': [
        'Native Support (past + future)',
        'Native Support (feat_dynamic_real)',
        'Supported (exogenous variables)',
        'XReg Support (separate install)',
        'Implicit (multivariate series)',
        'Not Explicitly Supported',
        'Exogenous Infusion',
        'Channel Independence',
        'Channel Mixing'
    ],
    
    'Multivariado': [
        'Yes',
        'Yes',
        'Multiple (independentes)',
        'Multiple (independentes)',
        'Yes (primary feature)',
        'Yes',
        'Yes',
        'Independent Channels',
        'Yes (correlated channels)'
    ],
    
    'Zero_Shot': [
        'Yes',
        'Yes',
        'Yes',
        'Yes',
        'Yes',
        'Yes',
        'Yes',
        'Pre-trained available',
        'Pre-trained available'
    ],
    
    'Casos_de_Uso_Ideais': [
        'Universal forecasting, Covariates',
        'Universal forecasting, Multitask',
        'Tabular time series, Fast inference',
        'Large scale forecasting, Research',
        'Observability metrics, Enterprise',
        'Multi-scale temporal, Lightweight',
        'Compact models, Fine-tuning',
        'Long-term forecasting, Channel-indep',
        'Multivariate correlated, Efficient'
    ],
    
    'Instalação': [
        'pip install chronos-forecasting',
        'pip install git+https://github.com/SalesforceAIResearch/uni2ts.git',
        'pip install tabpfn-time-series',
        'git clone https://github.com/google-research/timesfm.git && pip install -e .',
        'pip install toto-ts',
        'pip install git+https://github.com/ibm-granite/granite-tsfm.git',
        'pip install git+https://github.com/ibm-granite/granite-tsfm.git',
        'pip install transformers',
        'pip install transformers'
    ],
    
    'Prós_Principais': [
        '• Native covariate support\n• SOTA performance\n• Multi-step quantiles',
        '• Universal transformer\n• Multiple model sizes\n• GIFT-Eval leader',
        '• Zero training required\n• Fast inference\n• Strong benchmarks',
        '• Large context (16k)\n• Google research\n• BigQuery integration',
        '• Observability focus\n• 2.36T tokens trained\n• Enterprise ready',
        '• Multi-scale invariant\n• Compact model\n• State-space approach',
        '• Ultra-compact (<1M)\n• Fine-tuning fast\n• Exogenous support',
        '• Long-term forecasting\n• Patching efficiency\n• Channel independence',
        '• MLP-Mixer efficiency\n• Channel correlation\n• Lightweight'
    ],
    
    'Contras_Principais': [
        '• Large model size\n• Requires good covariates',
        '• Complex installation\n• GitHub dependency',
        '• Client-server model\n• Less control',
        '• Covariates require XReg install\n• PyTorch only',
        '• Observability focused\n• Limited to this domain',
        '• Scale factor tuning required\n• Limited examples',
        '• Very small, may lack expressiveness\n• Channel mixing optional',
        '• Channel independence may miss correlations\n• Transformer complexity',
        '• Less established than others\n• Limited to multivariate'
    ]
}

# Criar DataFrame de comparação
df_comparison = pd.DataFrame(models_comparison)

# Informações sobre covariáveis dinâmicas por modelo
covariate_details = {
    'Amazon Chronos-2': {
        'Método': 'Parâmetros future_df e context_df',
        'Como_usar': 'pipeline.predict_df(context_df, future_df=future_df, ...)',
        'Tipos_suportados': 'Past covariates, Future known covariates',
        'Exemplo': 'Preço da energia futuro, Temperatura passada'
    },
    
    'Salesforce Moirai-2.0': {
        'Método': 'feat_dynamic_real_dim e past_feat_dynamic_real_dim',
        'Como_usar': 'Moirai2Forecast(..., feat_dynamic_real_dim=N, past_feat_dynamic_real_dim=M)',
        'Tipos_suportados': 'Dynamic real features, Past dynamic features',
        'Exemplo': 'Variáveis exógenas futuras e passadas'
    },
    
    'TabPFN-TS': {
        'Método': 'Exogenous variables integration',
        'Como_usar': 'Incluir na matriz de features antes do forecasting',
        'Tipos_suportados': 'Variáveis exógenas quaisquer',
        'Exemplo': 'Preço, Temperatura, Promoção'
    },
    
    'Google TimesFM-2.5': {
        'Método': 'XReg (separate installation)',
        'Como_usar': 'uv pip install -e .[xreg] + forecast com covariates',
        'Tipos_suportados': 'External regressors',
        'Exemplo': 'Requiere instalação especial'
    },
    
    'Datadog Toto': {
        'Método': 'Multivariate series input',
        'Como_usar': 'MaskedTimeseries com múltiplas séries',
        'Tipos_suportados': 'Múltiplas séries temporais simultâneas',
        'Exemplo': 'CPU, Memory, Request rate simultaneamente'
    },
    
    'IBM Flowstate': {
        'Método': 'Não explícito',
        'Como_usar': 'Foco em escala temporal vs covariáveis',
        'Tipos_suportados': 'Não especificado na documentação',
        'Exemplo': 'Ajuste de scale_factor para diferentes frequências'
    },
    
    'IBM TTM': {
        'Método': 'Exogenous Infusion',
        'Como_usar': 'past_exogenous_cols e future_exogenous_cols',
        'Tipos_suportados': 'Exogenous variables, Static categorical',
        'Exemplo': 'Preço futuro, temperatura passada, tipo de loja'
    },
    
    'IBM PatchTST': {
        'Método': 'Channel Independence',
        'Como_usar': 'Cada canal como série univariada independente',
        'Tipos_suportados': 'Canais independentes (não covariáveis externas)',
        'Exemplo': 'Temperature, Humidity, Pressure como canais separados'
    },
    
    'IBM PatchTSMixer': {
        'Método': 'Channel Mixing',
        'Como_usar': 'Múltiplos canais correlacionados',
        'Tipos_suportados': 'Canais correlacionados (não externas)',
        'Exemplo': 'HUFL, HULL, MUFL correlacionados no ETTh1'
    }
}

def print_comparison_table():
    """Imprime tabela de comparação dos modelos"""
    print("="*120)
    print("TABELA COMPARATIVA DOS MODELOS DE FOUNDATION PARA TIME SERIES FORECASTING")
    print("="*120)
    
    for i, row in df_comparison.iterrows():
        print(f"\n{row['Modelo']}")
        print(f"  Parâmetros: {row['Parâmetros']}")
        print(f"  Arquitetura: {row['Arquitetura']}")
        print(f"  Covariáveis Dinâmicas: {row['Covariáveis_Dinâmicas']}")
        print(f"  Multivariado: {row['Multivariado']}")
        print(f"  Zero-Shot: {row['Zero_Shot']}")
        print(f"  Casos de Uso Ideais: {row['Casos_de_Uso_Ideais']}")
        print(f"  Instalação: {row['Instalação']}")
        print(f"  Prós: {row['Prós_Principais']}")
        print(f"  Contras: {row['Contras_Principais']}")

def print_covariate_details():
    """Imprime detalhes sobre covariáveis dinâmicas"""
    print("\n" + "="*120)
    print("DETALHES SOBRE SUPORTE A COVARIÁVEIS DINÂMICAS")
    print("="*120)
    
    for model, details in covariate_details.items():
        print(f"\n{model}:")
        print(f"  Método: {details['Método']}")
        print(f"  Como Usar: {details['Como_usar']}")
        print(f"  Tipos Suportados: {details['Tipos_suportados']}")
        print(f"  Exemplo: {details['Exemplo']}")

def print_model_selection_guide():
    """Imprime guia de seleção de modelo"""
    print("\n" + "="*120)
    print("GUIA DE SELEÇÃO DE MODELO")
    print("="*120)
    
    scenarios = {
        "Forecast com Covariáveis Futuras Conhecidas": [
            "Amazon Chronos-2 (Native future_df)",
            "Salesforce Moirai-2.0 (feat_dynamic_real)",
            "IBM TTM (Exogenous Infusion)"
        ],
        
        "Séries Multivariadas com Correlação": [
            "Datadog Toto (Primary feature)",
            "IBM PatchTSMixer (Channel Mixing)",
            "IBM TTM (Channel Mix)"
        ],
        
        "Modelos Compactos (< 1M parâmetros)": [
            "IBM TTM (< 1M)",
            "IBM Flowstate (< 10M)",
            "IBM PatchTST (616k)",
            "IBM PatchTSMixer (196k)"
        ],
        
        "Observabilidade e Métricas de Sistema": [
            "Datadog Toto (Primary focus)",
            "Amazon Chronos-2 (General purpose)",
            "Google TimesFM-2.5 (Research)"
        ],
        
        "Zero-Shot sem Treinamento": [
            "Amazon Chronos-2",
            "Salesforce Moirai-2.0", 
            "Google TimesFM-2.5",
            "TabPFN-TS",
            "IBM Flowstate"
        ],
        
        "Fine-tuning Rápido": [
            "IBM TTM (Minutes)",
            "IBM PatchTST",
            "IBM PatchTSMixer"
        ]
    }
    
    for scenario, models in scenarios.items():
        print(f"\n{scenario}:")
        for model in models:
            print(f"  • {model}")

def print_installation_guide():
    """Imprime guia de instalação"""
    print("\n" + "="*120)
    print("GUIA DE INSTALAÇÃO")
    print("="*120)
    
    installations = {
        "Amazon Chronos-2": "pip install chronos-forecasting",
        "Salesforce Moirai-2.0": "pip install git+https://github.com/SalesforceAIResearch/uni2ts.git",
        "TabPFN-TS": "pip install tabpfn-time-series",
        "Google TimesFM-2.5": "git clone https://github.com/google-research/timesfm.git && cd timesfm && pip install -e .",
        "Google TimesFM-2.5 (com XReg)": "cd timesfm && uv pip install -e .[xreg]",
        "Datadog Toto": "pip install toto-ts",
        "IBM Flowstate": "pip install git+https://github.com/ibm-granite/granite-tsfm.git",
        "IBM TTM": "pip install git+https://github.com/ibm-granite/granite-tsfm.git",
        "IBM PatchTST": "pip install transformers",
        "IBM PatchTSMixer": "pip install transformers"
    }
    
    for model, command in installations.items():
        print(f"\n{model}:")
        print(f"  {command}")

def create_benchmark_summary():
    """Cria resumo de benchmarks"""
    print("\n" + "="*120)
    print("RESUMO DE BENCHMARKS")
    print("="*120)
    
    benchmarks = {
        "GIFT-Eval (General Time Series Forecasting)": {
            "Top 3": [
                "1. TabPFN-TS (1st place as of Jan 2025)",
                "2. IBM Flowstate (Best Zero-Shot)",
                "3. Amazon Chronos-2 (SOTA performance)"
            ],
            "Observações": "TabPFN-TS lidera, mas requer cliente-servidor"
        },
        
        "BOOM (Observability Metrics)": {
            "Top 3": [
                "1. Datadog Toto (State-of-the-art)",
                "2. Amazon Chronos-2",
                "3. Google TimesFM-2.5"
            ],
            "Observações": "Toto é especializado em observabilidade"
        },
        
        "Chronos Benchmark II": {
            "Top 3": [
                "1. Amazon Chronos-2 (>90% win rate vs Bolt)",
                "2. Salesforce Moirai-2.0",
                "3. IBM TTM"
            ],
            "Observações": "Chronos-2 demonstra superioridade consistente"
        }
    }
    
    for benchmark, data in benchmarks.items():
        print(f"\n{benchmark}:")
        print(f"  Top 3:")
        for rank in data["Top 3"]:
            print(f"    {rank}")
        print(f"  Observações: {data['Observações']}")

if __name__ == "__main__":
    print_comparison_table()
    print_covariate_details()
    print_model_selection_guide()
    print_installation_guide()
    create_benchmark_summary()
    
    print("\n" + "="*120)
    print("CONCLUSÃO")
    print("="*120)
    print("""
Os 9 modelos apresentados representam o estado da arte em foundation models
para time series forecasting, cada um com especialidades distintas:

• Amazon Chronos-2: Melhor para forecasting universal com covariáveis
• Salesforce Moirai-2.0: Excelente para multitask e múltiplas escalas  
• TabPFN-TS: Líder em benchmarks, ideal para inferência rápida
• Google TimesFM-2.5: Forte em pesquisa e integração com Google Cloud
• Datadog Toto: Especializado em observabilidade e métricas enterprise
• IBM Flowstate: inovador com ajuste automático de escala temporal
• IBM TTM: solução compacta ideal para deployment em produção
• IBM PatchTST: eficiente para forecasting de longo prazo
• IBM PatchTSMixer: otimizado para séries correlacionadas

A escolha do modelo deve considerar:
1. Necessidade de covariáveis dinâmicas
2. Natureza dos dados (univariado vs multivariado)
3. Restrições de recursos computacionais
4. Requisitos de latência e eficiência
5. Domínio específico de aplicação
""")