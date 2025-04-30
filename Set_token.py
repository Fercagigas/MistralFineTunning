import os

# Establecer el token de Hugging Face
hf_token = "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"  # Reemplaza con tu token real

# Establecer el token como una variable de entorno
os.environ["HF_TOKEN_MISTRAL"] = hf_token

# Verificar que se haya establecido correctamente
print("Token de Hugging Face establecido:", os.environ.get("HF_TOKEN_MISTRAL"))
