import kagglehub

# download latest version
path = kagglehub.model_download("tensorflow/bert/tensorFlow2/bert-en-uncased-l-4-h-128-a-2")

print(f"Model downloaded to: {path}")