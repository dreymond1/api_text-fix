from flask import Flask, request

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

def carregar_modelo_e_tokenizer():
    model = load_model("files/sentiment_model.h5")  # Caminho para o modelo treinado
    with open("files/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("files/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    
    if not model or not tokenizer or not label_encoder:
        raise ValueError("Erro ao carregar modelo, tokenizer ou label_encoder. Verifique os arquivos!")
    
    return model, tokenizer, label_encoder

def substituir_termos(texto):
    for termo, substituto in substituicoes.items():
        texto = re.sub(termo, substituto, texto, flags=re.IGNORECASE)
    return texto

def testar_comentarios_dataframe(texto, model, tokenizer, max_len_contexto=50):
    
    data_base = substituir_termos(texto)
    
    # Tokenização dos comentários da coluna
    X_novos_comentarios = tokenizer.texts_to_sequences(data_base.tolist())
    
    # Padding para garantir o mesmo tamanho
    X_novos_comentarios = pad_sequences(X_novos_comentarios, maxlen=max_len_contexto, padding='post')
    
    # Previsão
    predicoes = model.predict(X_novos_comentarios)
    
    # Decodificar as previsões
    y_pred = np.argmax(predicoes, axis=1)  # Pega a classe com maior probabilidade
    return y_pred

# Função para mapear a previsão de volta ao sentimento original
def mapear_sentimento(predicoes_codificadas, label_encoder):
    sentimentos_preditos = label_encoder.inverse_transform(predicoes_codificadas)
    return sentimentos_preditos

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def main():
    data = request.json
    sentiment_pred_code = testar_comentarios_dataframe(data, model, tokenizer)
    predicted_sentiment = mapear_sentimento(sentiment_pred_code, label_encoder)[0]
    return predicted_sentiment

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
