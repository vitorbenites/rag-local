import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = os.getenv('CHROMA_PATH', 'chroma')
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'local-rag')
TEXT_EMBEDDING_MODEL = os.getenv('TEXT_EMBEDDING_MODEL', 'nomic-embed-text')


def get_vector_db():
    '''
    Função para retornar o database vetorial chroma.
    '''
    embedding = OllamaEmbeddings(
        model=TEXT_EMBEDDING_MODEL, show_progress=True)
    db = Chroma(collection_name=COLLECTION_NAME,
                persist_directory=CHROMA_PATH, embedding_function=embedding)
    return db
