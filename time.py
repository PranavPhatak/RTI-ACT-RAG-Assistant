from langchain_nvidia_ai_endpoints import ChatNVIDIA

models = ChatNVIDIA.get_available_models()

for model in models:
    print(model)