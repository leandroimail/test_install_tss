# Zero‑shot time‑series foundation models with dynamic covariates

This report documents how to use nine popular foundation models for zero‑shot time‑series forecasting when **dynamic covariates** (also called exogenous variables) are available. The goal was to research the official model cards, GitHub repositories and technical blogs to understand each model’s capabilities and to provide Python examples for forecasting with covariates. When the native interfaces do not explicitly support exogenous inputs, the examples show how to include covariates as additional channels. The attached covariate\_examples.py file contains ready‑to‑run code snippets for each model.

## Research summary

### Amazon Chronos‑2

Chronos‑2 is part of Amazon’s **Chronos** family of time‑series foundation models. The model card describes a Chronos2Pipeline class with a predict\_df method that supports past and future covariates. A data frame with the historical target values and covariate columns (both numerical and categorical) is passed as context\_df, and optional future covariate values are supplied in future\_df together with the forecast horizon. The method returns probabilistic forecasts at specified quantile levels. The model card makes clear that Chronos‑2 supports covariate‑informed tasks[\[1\]](https://huggingface.co/amazon/chronos-2#:~:text=Usage).

### Salesforce Moirai (1.1‑R‑large and 2.0‑R‑small)

The **Moirai** models are provided via the uni2ts library. The documented example shows how to convert a pandas DataFrame into a PandasDataset by specifying the list of dynamic real covariate columns (e.g., feat\_dynamic\_real=\["temp", "is\_holiday"\]). When instantiating the MoiraiForecast class for the 1.1‑R‑large model, the number of dynamic covariates must be passed as feat\_dynamic\_real\_dim (and past\_feat\_dynamic\_real\_dim if covariates are available only historically). The blog post explains that without setting these dimensions the model will ignore the covariates[\[2\]](https://www.datasciencewithmarco.com/blog/hands-on-with-moirai-a-foundation-forecasting-model-by-salesforce#:~:text=Moirai%20expects%20a%20,PandasDataset). For the 2.0‑R‑small model the covariate dimensions are zero because this version does not yet support dynamic covariates.

### TabPFN‑TS

TabPFN‑TS frames univariate forecasting as a tabular regression problem. Its README emphasises that exogenous variables can be incorporated seamlessly[\[3\]](https://raw.githubusercontent.com/PriorLabs/tabpfn-time-series/main/README.md#:~:text=,them%20into%20the%20forecasting%20model). Scripts in the repository show how to build a FeatureTransformer comprising **running index**, **calendar** and **auto‑seasonal** features, and then concatenate additional covariates before passing the tabular data to the pre‑trained model[\[4\]](https://raw.githubusercontent.com/PriorLabs/tabpfn-time-series/main/examples/sklearn_model_as_backbone.py#:~:text=,)[\[5\]](https://raw.githubusercontent.com/PriorLabs/tabpfn-time-series/main/examples/tabpfn_family_model_as_backbone.py#:~:text=feature_transformer%20%3D%20FeatureTransformer%28%20,). Zero‑shot inference uses TabPFNZeroShotRegressor which requires no training.

### Google TimesFM 2.5

TimesFM is a transformer‑based time‑series foundation model. An updated release adds an external regressors module (xreg) that allows **forecasting with covariates**. A Databricks blog explains the workflow: set up the environment, load a checkpoint, prepare the dataset and call forecast\_with\_covariates. The covariate input is passed via dictionaries for dynamic numerical/categorical and static numerical/categorical variables. An example predicts pollution while using temperature as a dynamic numerical covariate and wind direction as a static categorical covariate[\[6\]](https://community.databricks.com/t5/technical-blog/genai-for-time-series-analysis-with-timesfm/ba-p/95507).

### Datadog Toto (TOT)

The **Toto** forecaster treats the time series as a multi‑channel signal. The model card shows how to load the pre‑trained model, construct a tensor of shape (channels, time\_steps) and call the forecaster. Because each channel is forecast independently, dynamic covariates can be incorporated as additional channels. The example in the model card uses a MaskedTimeseries object and illustrates calling forecast[\[7\]](https://huggingface.co/Datadog/Toto-Open-Base-1.0/blob/349d00a43a03b0e9dc87af230f8dd9052c0de094/README.md#:~:text=from%20inference,toto%20import%20Toto).

### IBM Flowstate

Flowstate is a transformer model for multivariate forecasting. The Hugging Face example loads the model via FlowStateForPrediction.from\_pretrained and requires a 3‑D tensor whose dimensions correspond to (context\_length, batch\_size, channels). A scale factor is passed to normalise the input. The example emphasises that you need to choose an appropriate scale factor based on your sampling rate[\[8\]](https://huggingface.co/ibm-granite/granite-timeseries-flowstate-r1#:~:text=FlowState%20can%20be%20used%20to,make%20predictions%20as%20follows). Covariates are included simply by adding more channels to the input tensor.

### IBM TinyTimeMixer (TTM)

TinyTimeMixer is a compact MLP‑Mixer‑style model for zero/few‑shot forecasting. The sktime documentation presents an example using the TinyTimeMixerForecaster class. A target series y and a pandas DataFrame X containing exogenous variables are passed to fit. During prediction, the future values of the exogenous variables are supplied again via X[\[9\]](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.ttm.TinyTimeMixerForecaster.html#:~:text=Example%20with%20exogenous%20variables%3A). TTM also introduces an **exogenous mixer** component which fuses exogenous signals during fine‑tuning (highlighted in the TTM research paper[\[10\]](https://arxiv.org/pdf/2401.03955#:~:text=and%20exogenous%20correlations,TTM%20outperforms%20existing%20popular%20benchmarks)).

### IBM PatchTST and PatchTSMixer

PatchTST is a channel‑independent transformer that divides each univariate series into patches before encoding them. Each channel shares the same weights[\[11\]](https://huggingface.co/ibm-granite/granite-timeseries-patchtst#:~:text=At%20a%20high%20level%20the,forecast%20via%20an%20appropriate%20head). The base implementation does not provide a dedicated interface for exogenous variables, but covariates can be included by treating them as additional channels. PatchTSMixer is a lightweight MLP‑Mixer model. Similar to PatchTST, exogenous variables are handled by concatenating covariate series to the input channels. At the time of writing there is an open feature request to provide formal support for categorical and numerical covariates, so this workaround remains the recommended approach.

## Example summary table

| Model | Covariate support (key points) | Implementation hint |
| :---- | :---- | :---- |
| **Chronos‑2** | Supports past & future real/categorical covariates via predict\_df[\[1\]](https://huggingface.co/amazon/chronos-2#:~:text=Usage) | Provide context\_df and future\_df with covariate columns |
| **Moirai‑1.1‑R‑large** | Accepts dynamic real covariates when their dimension is specified[\[2\]](https://www.datasciencewithmarco.com/blog/hands-on-with-moirai-a-foundation-forecasting-model-by-salesforce#:~:text=Moirai%20expects%20a%20,PandasDataset) | Set feat\_dynamic\_real\_dim and past\_feat\_dynamic\_real\_dim; use PandasDataset |
| **Moirai‑2.0‑R‑small** | Current version does not use covariates | Set covariate dimensions to zero |
| **TabPFN‑TS** | Designed to include exogenous variables[\[3\]](https://raw.githubusercontent.com/PriorLabs/tabpfn-time-series/main/README.md#:~:text=,them%20into%20the%20forecasting%20model) | Engineer features (index, calendar, seasonal) and append exogenous columns |
| **TimesFM 2.5** | forecast\_with\_covariates accepts dynamic & static covariates[\[6\]](https://community.databricks.com/t5/technical-blog/genai-for-time-series-analysis-with-timesfm/ba-p/95507) | Supply covariate dictionaries and choose xreg\_mode |
| **Datadog Toto** | Multi‑channel input; extra channels can hold covariates[\[7\]](https://huggingface.co/Datadog/Toto-Open-Base-1.0/blob/349d00a43a03b0e9dc87af230f8dd9052c0de094/README.md#:~:text=from%20inference,toto%20import%20Toto) | Stack covariate series with target into (channels, time\_steps) tensor |
| **IBM Flowstate** | 3‑D input (context\_length, batch\_size, channels); additional channels for covariates[\[8\]](https://huggingface.co/ibm-granite/granite-timeseries-flowstate-r1#:~:text=FlowState%20can%20be%20used%20to,make%20predictions%20as%20follows) | Normalise using scale\_factor and include covariates as channels |
| **IBM TinyTimeMixer** | Fits on target series with exogenous DataFrame X[\[9\]](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.ttm.TinyTimeMixerForecaster.html#:~:text=Example%20with%20exogenous%20variables%3A) | Call fit(y, X) and predict(fh, X) |
| **IBM PatchTST / PatchTSMixer** | Channel‑independent models; no native exogenous interface[\[11\]](https://huggingface.co/ibm-granite/granite-timeseries-patchtst#:~:text=At%20a%20high%20level%20the,forecast%20via%20an%20appropriate%20head) | Concatenate covariates as additional input channels |

## How to run the examples

1\.    	Install the required Python libraries for the models you intend to try. Many of these libraries (chronos‑forecasting, uni2ts, tabpfn\_ts, timesfm, tot, tsfm\_public, sktime, transformers) are available via pip but may require specific versions or GPU support.

2\.    	Download the script covariate\_examples.py. Each function in the script corresponds to one model and prepares a synthetic dataset with a target and one or more covariates.

3\.    	Uncomment the function you wish to run in the if \_\_name\_\_ \== "\_\_main\_\_" block. Ensure that your environment has enough memory and the necessary libraries. Some functions may download large model weights the first time they are called.

These examples provide a starting point for applying state‑of‑the‑art foundation models to practical forecasting problems where additional signals (e.g., weather, calendar or economic indicators) influence the future evolution of the target series.

---

[\[1\]](https://huggingface.co/amazon/chronos-2#:~:text=Usage) amazon/chronos-2 · Hugging Face

[https://huggingface.co/amazon/chronos-2](https://huggingface.co/amazon/chronos-2)

[\[2\]](https://www.datasciencewithmarco.com/blog/hands-on-with-moirai-a-foundation-forecasting-model-by-salesforce#:~:text=Moirai%20expects%20a%20,PandasDataset)  Hands-On with Moirai: A Foundation Forecasting Model by Salesforce

[https://www.datasciencewithmarco.com/blog/hands-on-with-moirai-a-foundation-forecasting-model-by-salesforce](https://www.datasciencewithmarco.com/blog/hands-on-with-moirai-a-foundation-forecasting-model-by-salesforce)

[\[3\]](https://raw.githubusercontent.com/PriorLabs/tabpfn-time-series/main/README.md#:~:text=,them%20into%20the%20forecasting%20model) raw.githubusercontent.com

[https://raw.githubusercontent.com/PriorLabs/tabpfn-time-series/main/README.md](https://raw.githubusercontent.com/PriorLabs/tabpfn-time-series/main/README.md)

[\[4\]](https://raw.githubusercontent.com/PriorLabs/tabpfn-time-series/main/examples/sklearn_model_as_backbone.py#:~:text=,) raw.githubusercontent.com

[https://raw.githubusercontent.com/PriorLabs/tabpfn-time-series/main/examples/sklearn\_model\_as\_backbone.py](https://raw.githubusercontent.com/PriorLabs/tabpfn-time-series/main/examples/sklearn_model_as_backbone.py)

[\[5\]](https://raw.githubusercontent.com/PriorLabs/tabpfn-time-series/main/examples/tabpfn_family_model_as_backbone.py#:~:text=feature_transformer%20%3D%20FeatureTransformer%28%20,) raw.githubusercontent.com

[https://raw.githubusercontent.com/PriorLabs/tabpfn-time-series/main/examples/tabpfn\_family\_model\_as\_backbone.py](https://raw.githubusercontent.com/PriorLabs/tabpfn-time-series/main/examples/tabpfn_family_model_as_backbone.py)

[\[6\]](https://community.databricks.com/t5/technical-blog/genai-for-time-series-analysis-with-timesfm/ba-p/95507) GenAI for Time Series Analysis with TimesFM \- Databricks Community \- 95507

[https://community.databricks.com/t5/technical-blog/genai-for-time-series-analysis-with-timesfm/ba-p/95507](https://community.databricks.com/t5/technical-blog/genai-for-time-series-analysis-with-timesfm/ba-p/95507)

[\[7\]](https://huggingface.co/Datadog/Toto-Open-Base-1.0/blob/349d00a43a03b0e9dc87af230f8dd9052c0de094/README.md#:~:text=from%20inference,toto%20import%20Toto) README.md · Datadog/Toto-Open-Base-1.0 at 349d00a43a03b0e9dc87af230f8dd9052c0de094

[https://huggingface.co/Datadog/Toto-Open-Base-1.0/blob/349d00a43a03b0e9dc87af230f8dd9052c0de094/README.md](https://huggingface.co/Datadog/Toto-Open-Base-1.0/blob/349d00a43a03b0e9dc87af230f8dd9052c0de094/README.md)

[\[8\]](https://huggingface.co/ibm-granite/granite-timeseries-flowstate-r1#:~:text=FlowState%20can%20be%20used%20to,make%20predictions%20as%20follows) ibm-granite/granite-timeseries-flowstate-r1 · Hugging Face

[https://huggingface.co/ibm-granite/granite-timeseries-flowstate-r1](https://huggingface.co/ibm-granite/granite-timeseries-flowstate-r1)

[\[9\]](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.ttm.TinyTimeMixerForecaster.html#:~:text=Example%20with%20exogenous%20variables%3A) TinyTimeMixerForecaster — sktime documentation

[https://www.sktime.net/en/latest/api\_reference/auto\_generated/sktime.forecasting.ttm.TinyTimeMixerForecaster.html](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.ttm.TinyTimeMixerForecaster.html)

[\[10\]](https://arxiv.org/pdf/2401.03955#:~:text=and%20exogenous%20correlations,TTM%20outperforms%20existing%20popular%20benchmarks) 2401.03955

[https://arxiv.org/pdf/2401.03955](https://arxiv.org/pdf/2401.03955)

[\[11\]](https://huggingface.co/ibm-granite/granite-timeseries-patchtst#:~:text=At%20a%20high%20level%20the,forecast%20via%20an%20appropriate%20head) ibm-granite/granite-timeseries-patchtst · Hugging Face

[https://huggingface.co/ibm-granite/granite-timeseries-patchtst](https://huggingface.co/ibm-granite/granite-timeseries-patchtst)

