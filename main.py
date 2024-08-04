import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, make_response
from flask_restx import Resource, Api, fields
from werkzeug.datastructures import FileStorage
from src.embed import embed
from src.query import query

# Carregar variáveis de ambiente
load_dotenv()

# Diretório temporário de arquivos
TMP_DIR = os.getenv('TEMP_DIR', './tmp')

# Cria o diretório temporário se ele não existir
os.makedirs(TMP_DIR, exist_ok=True)

# Declaração do flask app
app = Flask(__name__)
api = Api(app, title='RAG Ollama Chroma',
          description='API para inserir documentos PDF e realizar queries' +
          ' ao LLM.',
          default='Endpoints',
          default_label='Post')

# Modelos
query_model = api.model('Fazer uma pergunta', {
    'system': fields.String(example='Você é um assistente de IA que fornece' +
                            ' respostas concisas e precisas baseado ' +
                            'em documentos.'),
    'prompt': fields.String(example='Faça uma pergunta'),
    'temperature': fields.Float(example=0.7)
})

# File upload
upload_parser = api.parser()
upload_parser.add_argument('file', location='files', type=FileStorage,
                           required=True, help='Arquivo para inserção')


@api.route("/embed")
class EmbedPost(Resource):
    @api.doc('Inserir arquivo no banco de dados vetorial.')
    @api.expect(upload_parser)
    def post(self):
        '''
        Inserção de arquivo no banco de dados vetorial.
        Método POST
        '''
        args = upload_parser.parse_args()
        file = args['file']

        if file is None:
            return make_response(jsonify({"error": "Sem arquivo."}), 400)

        if file.filename == '':
            return make_response(jsonify({"error": "Arquivo não selecionado."}), 400)

        embedded = embed(file)

        if embedded:
            return make_response(jsonify({"message": "Arquivo inserido com sucesso."}), 200)

        return make_response(jsonify({"message": "Arquivo não foi inserido."}), 400)


@api.route('/query')
class Query(Resource):
    @api.doc('Perguntar ao LLM.')
    @api.expect(query_model, validate=True)
    def post(self):
        '''
        Faz uma query ao llm e retorna a resposta.
        Método post
        '''
        data = request.get_json()
        response = query(system=data.get('system'), input=data.get(
            'prompt'), temperature=data.get('temperature'))
        if response:
            return make_response(jsonify({"message": response}), 200)
        return make_response(jsonify({"error": "Algo deu errado."}), 400)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050, debug=True)
