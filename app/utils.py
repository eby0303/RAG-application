from langchain_community.embeddings import LlamaCppEmbeddings

def get_embeddings():
    return LlamaCppEmbeddings(
        model_path="models/e5-small-v2.Q4_K_M.gguf",
    )