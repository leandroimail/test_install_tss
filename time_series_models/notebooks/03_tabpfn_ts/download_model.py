import os
from tabpfn.model.loading import resolve_model_path, download_model

print("Iniciando o download do modelo TabPFN...")

# Usar um dos nomes de modelo válidos da mensagem de erro
model_name = "tabpfn-v2-regressor.ckpt"

# Resolve o caminho do modelo
model_path, _, model_name, which = resolve_model_path(
    model_name,
    which="regressor",
)

if not model_path.exists():
    print(f"Modelo não encontrado em {model_path}. Fazendo o download...")
    download_model(
        to=model_path,
        which=which,
        version="v2",
        model_name=model_name,
    )
    print("Download do modelo concluído.")
else:
    print("O modelo já existe localmente.")
