import os
from datetime import datetime
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from get_vector_db import get_vector_db

TMP_DIR = os.getenv('TMP_DIR', './tmp')


def allowed_file(filename):
    '''
    Função para verificar se o arquivo é válido (.pdf)
    '''
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}


def save_file(file):
    '''
    Função para salvar o arquivo no diretório temporário.
    '''
    current_time = datetime.now()
    timestamp = current_time.timestamp()
    filename = str(timestamp) + "_" + secure_filename(file.filename)
    file_path = os.path.join(TMP_DIR, filename)
    file.save(file_path)
    return file_path


def load_and_split_data(file_path):
    '''
    Função para carregar e dividir os dados do arquivo pdf.
    '''
    loader = UnstructuredPDFLoader(file_path=file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    chunks = filter_complex_metadata(chunks)
    return chunks


def embed(file):
    '''
    Função principal para manipular a inserção de arquivos.
    '''
    if file.filename != '' and file and allowed_file(file.filename):
        file_path = save_file(file)
        chunks = load_and_split_data(file_path)
        db = get_vector_db()
        db.add_documents(chunks)
        os.remove(file_path)
        return True
    return False
