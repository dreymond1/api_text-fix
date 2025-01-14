from flask import Flask, request, jsonify
import re
import numpy as np
import gc


app = Flask(__name__)

# Dicionário de substituições
substituicoes = {
    r'\bnao\b': 'não',
    r'\bn\b': 'não',
    r'\brapido\b': 'rápido',
    r'\brapida\b': 'rápida',
    r'\brapidos\b': 'rápidos',
    r'\brapidas\b': 'rápidas',
    r'\bvoce\b': 'você',
    r'\bvoces\b': 'vocês',
    r'\bvc\b': 'você',
    r'\bvcs\b': 'vocês',
    r'\bcomentario\b': 'comentário',
    r'\bcomentarios\b': 'comentários',
    r'\bq\b': 'que',
    r'\bobg\b': 'obrigado',
    r'\bola\b': 'olá',
    r'\bfacil\b': 'fácil',
    r'\bdificil\b': 'difícil',
    r'\bdificeis\b': 'difíceis',
    r'\bagente\b': 'a gente',
    r'\bpratico\b': 'prático',
    r'\bpratica\b': 'prática',
    r'\bpq\b': 'por que',
    r'\bso\b': 'só',
    r'\bsao\b': 'são',
    r'\bagil\b': 'ágil',
    r'\bpessimo\b': 'péssimo',
    r'\bpessima\b': 'péssima',
    r'\bpessimos\b': 'péssimos',
    r'\bpessimas\b': 'péssimas',
    r'\bhorrivelll\b': 'horrível',
    r'\bhorivel\b': 'horrível',
    r'\bhorriveis\b': 'horríveis',
    r'\bhorrivel\b': 'horrível',
    r'\bagil\b': 'ágil',
    r'\bsimpatica\b': 'simpática',
    r'\binfelismente\b': 'infelizmente',
    r'\bproblematico\b': 'problemático',
    r'\bdecepçao\b': 'decepção',
    r'\binsuportavel\b': 'insuportável',
    r'\binsuportaveis\b': 'insuportáveis',
    r'\birreparavel\b': 'irreparável',
    r'\bindisponivel\b': 'indisponível',
    r'\bantipatico\b': 'antipático',
    r'\botimo\b': 'ótimo',
    r'\botimos\b': 'ótimos',
    r'\botima\b': 'ótima',
    r'\botimas\b': 'ótimas',
    r'\bageis\b': 'ágeis',
    r'\bfacil\b': 'fácil',
    r'\bfaceis\b': 'fáceis',
    r'\bincrivel\b': 'incrível',
    r'\bincriveis\b': 'incríveis',
    r'\bfantastico\b': 'fantástico',
    r'\bfantastica\b': 'fantástica',
    r'\bfantasticos\b': 'fantásticos',
    r'\bfantasticas\b': 'fantásticas',
    r'\bsatisfatorio\b': 'satisfatório',
    r'\bsatisfatoria\b': 'satisfatória',
    r'\bsatisfatorios\b': 'satisfatórios',
    r'\bsatisfatorias\b': 'satisfatórias',
    r'\bimpecavel\b': 'impecável',
    r'\bimpecaveis\b': 'impecáveis',
    r'\bvalera\b': 'valerá',
    r'\butil\b': 'útil',
    r'\binsatisfatoriob\b': 'insatisfatório',
    r'\binsatisfatoria\b': 'insatisfatória',
    r'\binsatisfatorios\b': 'insatisfatórios',
    r'\binsatisfatorias\b': 'insatisfatórias',
    r'\bdesconfortavel\b': 'desconfortável',
    r'\bdesconfortsveis\b': 'desconfortáveis',
    r'\bprejuizo\b': 'prejuízo',
    r'\bpreju\b': 'prejuízo',
    r'\bdificeis\b': 'difíceis',
    r'\bunico\b': 'único',
    r'\bunica\b': 'única',
    r'\bunicas\b': 'únicas',
    r'\bunicos\b': 'únicos',
    r'\bmais rapido\b': 'mais rápido',
    r'\bmais agil\b': 'mais ágil'
}

logging.basicConfig(level=logging.DEBUG)

def substituir_termos(texto):
    texto = texto[:1000]  # Limite para evitar uso excessivo de memória
    for termo, substituto in substituicoes.items():
        texto = re.sub(termo, substituto, texto, flags=re.IGNORECASE)
    return texto


@app.route("/", methods=["POST"])
def predict():
    try:
        dados = request.json
        texto = dados.get("texto", "")
        
        # Log para ver o que estamos recebendo
        app.logger.debug(f"Texto recebido: {texto}")

        if not texto:
            app.logger.error("Texto não fornecido.")
            return jsonify({"error": "Texto não fornecido"}), 400

        # Processamento
        textos = [texto]  # Para o caso de ser um único texto, podemos colocar em uma lista
        resultados = [substituir_termos(t) for t in textos]  # Aplica substituições para cada texto
        app.logger.debug(f"Texto após substituições: {resultados}")

        # Libere memória
        gc.collect()

        return jsonify(resultados)  # Retorna como lista de textos processados
    except Exception as e:
        # Log de erro detalhado
        app.logger.error(f"Erro ao processar o texto: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
