import os
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from get_vector_db import get_vector_db

LLM_MODEL = os.getenv('LLM_MODEL', 'llama3.1')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')


def get_prompt():
    '''
    Função para gerar o prompt melhorado.
    '''
    template = """
    Responda a pergunta baseado APENAS no contexto abaixo:
    {context}
    Pergunta: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt


def query(input,
          system="Você é um assistente de IA que fornece \
          resposta concisas e precisas",
          temperature=0.7):
    '''
    Função principal para lidar com o processo de consulta.
    '''
    if input:
        # Inicializa o modelo
        llm = ChatOllama(model=LLM_MODEL, system=system,
                         temperature=temperature, base_url=OLLAMA_URL)
        # Obtém a instancia do banco de dados vetorial
        db = get_vector_db()
        # Obtém o template do prompt
        prompt_template = get_prompt()

        # Recupera os documentos relevantes do banco de dados vetorial
        retriever = db.as_retriever()

        # Chain para gerar resposta
        chain = ({
            "context": retriever,
            "question": RunnablePassthrough()
        }
            | prompt_template
            | llm
            | StrOutputParser()
        )
        response = chain.invoke(input)
        # Retorno da resposta
        return response
    return None
