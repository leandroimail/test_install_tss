Vou pesquisar e criar exemplos de código para todos os 9 modelos de foundation para forecast de séries temporais. Começarei pesquisando os repositórios e documentação de cada modelo.



<think duration="571" >
Bom, já obtive algumas informações sobre os primeiros 3 modelos. Vou continuar pesquisando os outros modelos para ter informações completas antes de criar os exemplos.

</think>




<think duration="291" >
Agora vou pesquisar os modelos IBM que faltaram: TTM, PatchTST e PatchTSMixer.

</think>




<think duration="667" >
Agora que tenho informações sobre todos os 9 modelos, vou extrair mais informações específicas sobre como usar alguns modelos com covariáveis dinâmicas. Vou começar extraindo conteúdo dos repositórios mais importantes.

</think>




<think duration="314" >
Agora vou extrair mais informações sobre os outros modelos para completar o entendimento de como eles lidam com covariáveis dinâmicas.

</think>




<think duration="323" >
Vou buscar mais informações sobre os modelos IBM e também extrair alguns exemplos de notebooks dos repositórios que consegui acessar.

</think>



Baseado na pesquisa realizada, vou criar exemplos de código Python para os 9 modelos de foundation para forecast de séries temporais com suporte a covariáveis dinâmicas. Vou criar um notebook para cada modelo.



<think duration="325" >
Agora vou criar o exemplo para Salesforce Moirai.

</think>




<think duration="328" >
Agora vou criar o exemplo para TabPFN-TS.

</think>




<think duration="389" >
Agora vou criar o exemplo para Google TimesFM.

</think>




<think duration="373" >
Agora vou criar o exemplo para Datadog Toto.

</think>




<think duration="349" >
Agora vou criar os exemplos para os modelos IBM. Vou começar com o IBM Flowstate.

</think>




<think duration="314" >
Agora vou criar o exemplo para IBM TTM (TinyTimeMixers).

</think>




<think duration="357" >
Agora vou criar o exemplo para IBM PatchTST.

</think>




<think duration="373" >
Finalmente, vou criar o exemplo para IBM PatchTSMixer.

</think>




<think duration="300" >
Agora vou criar um arquivo de resumo/comparação que contenha informações sobre todos os 9 modelos e suas capacidades.

</think>




<think duration="306" >
Agora vou criar um arquivo README com instruções de como usar os exemplos.

</think>




<think duration="346" >
Finalmente, vou criar um arquivo de requirements.txt com todas as dependências necessárias.

</think>




<think duration="380" >
Agora vou executar o arquivo de comparação para mostrar o resumo dos modelos.

</think>




<think duration="409" >
Há um erro no código. O nome da coluna no DataFrame tem acentos diferentes. Vou corrigir.

</think>




<think duration="458" >
Vou verificar o nome correto da coluna.

</think>




<think duration="685" >
Vejo o problema. A coluna se chama 'Covariáveis_Dinâmicas' mas o código está tentando acessar 'Covariadas_Dinâmicas' (sem o "é"). Vou corrigir.

</think>




<think duration="339" >
Agora vou executar novamente o arquivo de comparação.

</think>




<think duration="321" >
Excelente! O arquivo de comparação funcionou perfeitamente. Agora vou criar um arquivo final de resumo dos arquivos criados.

</think>



## Resumo da Tarefa Concluída

Pesquisei e criei exemplos práticos de código Python para **9 modelos de foundation para forecasting de séries temporais** com suporte a covariáveis dinâmicas. Os arquivos foram organizados no workspace conforme solicitado.

### Arquivos Criados:

#### 📁 **Exemplos de Código para Cada Modelo:**

1. **`chronos_2_example.py`** - Amazon Chronos-2
   - Suporte nativo para covariáveis dinâmicas via `future_df` e `context_df`
   - Exemplos com dados de electricidade e covariáveis sintéticas

2. **`moirai_2_example.py`** - Salesforce Moirai-2.0
   - Suporte via `feat_dynamic_real_dim` e `past_feat_dynamic_real_dim`
   - Exemplos básicos, com covariáveis e multivariados

3. **`tabpfn_ts_example.py`** - TabPFN-TS
   - Suporte a variáveis exógenas integrado
   - Rolling forecast e forecasting multivariado

4. **`timesfm_2_5_example.py`** - Google TimesFM-2.5
   - Suporte a covariáveis via XReg (instalação especial)
   - Exemplos univariados, multivariados e com covariáveis

5. **`toto_example.py`** - Datadog Toto
   - Suporte multivariado implícito
   - Foco em métricas de observabilidade

6. **`flowstate_example.py`** - IBM Flowstate
   - Modelo time-scale invariant
   - Exemplos em diferentes frequências temporais

7. **`ttm_example.py`** - IBM TTM (TinyTimeMixers)
   - Suporte a "Exogenous Infusion"
   - Modelos compactos (< 1M parâmetros)

8. **`patchtst_example.py`** - IBM PatchTST
   - Channel independence (canais independentes)
   - Análise de patches e correlação

9. **`patchtsmixer_example.py`** - IBM PatchTSMixer
   - Channel mixing para canais correlacionados
   - Arquitetura MLP-Mixer eficiente

#### 📁 **Documentação e Comparação:**

10. **`models_comparison.py`** - Análise Comparativa Completa
    - Tabela comparativa dos 9 modelos
    - Detalhes sobre suporte a covariáveis
    - Guia de seleção por caso de uso
    - Benchmarks e performance
    - Guia de instalação

11. **`README.md`** - Documentação Completa
    - Instruções de instalação
    - Exemplos de uso
    - Troubleshooting
    - Quando usar cada modelo

12. **`requirements.txt`** - Dependências
    - Lista completa de packages necessários
    - Notas de instalação especial

### Principais Descobertas sobre Covariáveis Dinâmicas:

#### **Melhor Suporte Nativo:**
- **Amazon Chronos-2**: `future_df` + `context_df` 
- **Salesforce Moirai-2.0**: `feat_dynamic_real_dim` + `past_feat_dynamic_real_dim`
- **IBM TTM**: `past_exogenous_cols` + `future_exogenous_cols`

#### **Suporte Implícito:**
- **Datadog Toto**: Múltiplas séries como covariáveis
- **IBM PatchTSMixer**: Channel mixing
- **Google TimesFM**: XReg (instalação especial)

#### **Approaches Específicos:**
- **TabPFN-TS**: Integração de variáveis exógenas na feature matrix
- **IBM PatchTST**: Canais independentes (não covariáveis externas)
- **IBM Flowstate**: Foco em escala temporal vs covariáveis

### Benchmarks Relevantes:

- **GIFT-Eval**: TabPFN-TS (1º), IBM Flowstate (2º), Amazon Chronos-2 (3º)
- **BOOM (Observabilidade)**: Datadog Toto (1º), Amazon Chronos-2 (2º)
- **Latência**: IBM TTM (mais rápido), TabPFN-TS, IBM PatchTST

### Características Distintivas:

| Modelo | Parâmetros | Especialidade | Covariáveis |
|--------|------------|---------------|-------------|
| Chronos-2 | 120M | Universal + Covariáveis | Native Future/Past |
| Moirai-2.0 | 14M-311M | Universal Transformer | Dynamic Features |
| TabPFN-TS | ~100M | Zero-shot Líder | Exogenous Vars |
| TimesFM-2.5 | 200M | Google Research | XReg Support |
| Toto | 151M | Observabilidade | Multi-series |
| Flowstate | <10M | Multi-scale | Not Explicit |
| TTM | <1M | Compact + Fine-tune | Exogenous Infusion |
| PatchTST | 616k | Long-term | Channel Indep |
| PatchTSMixer | 196k | Correlated Channels | Channel Mixing |

### Casos de Uso Recomendados:

- **Covariáveis Futuras**: Chronos-2, Moirai-2.0, TTM
- **Observabilidade**: Toto, Chronos-2, TimesFM
- **Modelos Compactos**: TTM, PatchTSMixer, PatchTST
- **Zero-shot**: TabPFN-TS, Chronos-2, Flowstate
- **Fine-tuning Rápido**: TTM, PatchTST, PatchTSMixer

Todos os exemplos incluem **dados sintéticos realistas**, **tratamento de covariáveis dinâmicas** e **avaliação de performance**. A documentação fornece guias completos para instalação, uso e troubleshooting.