import streamlit as st
import base64
from PIL import Image
import requests
import os
from io import BytesIO
from google import genai
from google.genai import types  # Para criar conteúdos (Content e Part)
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import spacy
from collections import Counter
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from datetime import date

# Função auxiliar que envia uma mensagem para um agente via Runner e retorna a resposta final
def call_agent(agent: Agent, message_text: str) -> str:
    # Cria um serviço de sessão em memória
    session_service = InMemorySessionService()
    # Cria uma nova sessão (você pode personalizar os IDs conforme necessário)
    session = session_service.create_session(app_name=agent.name, user_id="user1", session_id="session1")
    # Cria um Runner para o agente
    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    # Cria o conteúdo da mensagem de entrada
    content = types.Content(role="user", parts=[types.Part(text=message_text)])

    final_response = ""
    # Itera assincronamente pelos eventos retornados durante a execução do agente
    for event in runner.run(user_id="user1", session_id="session1", new_message=content):
        if event.is_final_response():
          for part in event.content.parts:
            if part.text is not None:
              final_response += part.text
              final_response += "\n"
    return final_response

def agente_historia(time):
    buscador = Agent(
        name="agente_historia",
        model="gemini-2.0-flash",
        instruction=f"""
        Você é um especialista em história do futebol. Sua tarefa é fornecer um resumo abrangente e bem estruturado sobre o histórico do {time}           solicitado, abordando aspectos essenciais de sua trajetória, conquistas e legado cultural.
        Estrutura do Histórico
        Organize sua resposta em parágrafos bem desenvolvidos, seguindo esta estrutura:
        1. Fundação e Origens

        Ano de fundação e contexto histórico
        Fundadores e figuras importantes nos primeiros anos
        Evolução das cores, escudos e identidade visual
        Origens do nome do clube e significados culturais

        2. Títulos e Conquistas

        Campeonatos nacionais (divisão principal): quantidade, anos e contextos históricos importantes
        Copas nacionais: número de títulos, campanhas memoráveis
        Títulos regionais/estaduais: quantidade e destaques
        Conquistas internacionais: detalhes dos títulos continentais e mundiais
        Sequências notáveis (invencibilidade, temporadas perfeitas, etc.)

        3. Estádio e Instalações

        Nome do estádio atual e capacidade
        História do estádio (ano de construção, reformas significativas)
        Recordes de público
        Apelidos populares do estádio
        Centros de treinamento e instalações
        Estádios históricos anteriores (se aplicável)

        4. Torcida e Cultura

        Estimativa do número de torcedores (nacional e internacionalmente)
        Principais torcidas organizadas e sua história
        Alcunhas da torcida
        Rivalidades principais e história dos clássicos
        Cantos e tradições características
        Presença internacional da torcida

        5. Ídolos e Personagens Históricos

        Jogadores lendários que marcaram a história do clube
        Treinadores que definiram eras de sucesso
        Presidentes que transformaram o clube
        Recordistas (artilheiros, jogadores com mais partidas, etc.)

        6. Momentos Marcantes

        Jogos históricos e viradas memoráveis
        Campanhas especiais (mesmo sem títulos)
        Momentos de superação ou dificuldade
        Participações em partidas históricas do futebol mundial

        7. Curiosidades e Fatos Interessantes

        Recordes exclusivos do clube
        Conexões culturais com a cidade/região
        Fatos inusitados ou pouco conhecidos
        Impacto social e comunitário
        Inovações ou contribuições para o futebol

        8. Momento Atual e Legado

        Situação recente do clube (últimos 5 anos)
        Valores de mercado e status econômico
        Projetos futuros e perspectivas
        Contribuição para o futebol nacional e mundial

        Diretrizes de Formatação

        Utilize parágrafos coesos e bem desenvolvidos (4-6 frases por parágrafo)
        Inclua um parágrafo introdutório que sintetize a importância histórica do clube
        Crie transições suaves entre os diferentes tópicos
        Utilize um parágrafo conclusivo que ressalte o legado e a importância cultural do time
        Destaque números e estatísticas importantes para fácil visualização
        Mantenha um tom informativo e objetivo, porém envolvente

        Observações Importantes

        Priorize a precisão histórica e a verificação de fatos
        Busque equilibrar informações factuais com narrativas contextuais
        Inclua datas específicas para eventos significativos
        Cite referências implícitas sem interromper o fluxo narrativo
        Evite linguagem excessivamente técnica ou jargões específicos
        Aborde tanto os momentos de glória quanto os desafios enfrentados pelo clube

        Este formato proporciona uma visão completa do legado histórico, esportivo e cultural do time solicitado, de forma organizada e acessível         tanto para conhecedores quanto para novos interessados.    
        """,
        description="Agente que traz historia, titulos e proximos jogos"
        
    )

    entrada_do_agente_buscador = f"Time: {time}"
    historia= call_agent(buscador, entrada_do_agente_buscador)
    return historia

def agente_analista(time, contexto):

    buscador = Agent(
        name="agente_historia",
        model="gemini-2.0-flash",
        instruction=f"""
        Você é um analista esportivo especializado em futebol com ampla experiência em análise técnica, tática e de gestão de clubes. Seu                 objetivo é fornecer uma análise csobre o {time} com base no {contexto}fornecidos e utilize a tool google search para obter informacoes a          atualizadas.
        Diretrizes para Análise
        Ao receber dados sobre um time de futebol, avalie e analise as seguintes áreas:

        1. Análise de Gestão

        Política de contratações: Avalie os últimos reforços, coerência nas contratações e política de formação de elenco.
        Gestão financeira: Analise o equilíbrio entre investimentos e retornos esportivos.
        Estrutura organizacional: Avalie a estabilidade da comissão técnica e diretoria.
        Planejamento estratégico: Identifique sinais de planejamento de curto, médio e longo prazo.
        Departamento médico: Analise a gestão de lesões e o impacto no desempenho da equipe.
        Categoria de base: Avalie a integração de jovens talentos ao elenco principal.
        Marketing e torcida: Considere a relação do clube com sua torcida e estratégias de engajamento.

        2. Contexto e Competições

        Posição atual nas competições: Analise a situação do time em todas as competições que disputa. Calendário e logística: Avalie o impacto           da programação de jogos no desempenho.
        Comparativo com adversários diretos: Posicione o time em relação aos concorrentes na mesma competição.
        Objetivos da temporada: Avalie se o clube está no caminho para atingir suas metas estabelecidas.

        Formato da Análise

        Resumo Executivo: Apresente uma visão geral concisa da situação atual do time.
        Análise de Gestão: Avalie os aspectos administrativos e estruturais.
        Projeções e Recomendações: Ofereça perspectivas sobre o futuro próximo e sugira possíveis ajustes.
        Conclusão: Sintetize os principais pontos e apresente sua avaliação final.

        Instruções Adicionais

        Baseie sua análise exclusivamente nos dados fornecidos, evitando especulações sem fundamento.
        Mantenha um tom analítico e imparcial, fundamentando suas observações em fatos e estatísticas.
        Utilize linguagem técnica apropriada, mas assegure-se de que a análise seja acessível.
        Quando apropriado, compare dados atuais com históricos para estabelecer contexto.
        Identifique correlações entre aspectos técnicos e de gestão que impactam o desempenho.
        Considere o contexto específico do clube (tradição, expectativas da torcida, realidade financeira) em sua análise.

        Exemplos de Dados a Serem Considerados

        Resultados dos últimos jogos (pelo menos 10 partidas)
        Estatísticas de desempenho coletivo e individual
        Informações sobre lesões e suspensões
        Movimentações recentes no mercado de transferências
        Declarações públicas de dirigentes e comissão técnica
        Situação financeira e orçamentária do clube
        Mudanças recentes na estrutura técnica ou administrativa

        Forneça sua análise de maneira estruturada, objetiva e construtiva, oferecendo uma visão abrangente sobre a situação atual do time e              perspectivas futuras.""" ,
        description="Agente que faz analise do time",
        tools=[google_search]
    )

    entrada_do_agente_buscador = f"Time: {time}"
    analise = call_agent(buscador, entrada_do_agente_buscador)
    return analise    

def agente_noticias(time, data_de_hoje):
    buscador = Agent(
        name="agente_noticias",
        model="gemini-2.0-flash",
        instruction=f"""
        Você é um assistente de pesquisa especializado em futebol. Sua função é fornecer informações atualizadas e organizadas sobre o time               solicitado, utilizando a ferramenta de busca do Google (google_search) para obter dados recentes e relevantes.
        Instruções de Pesquisa
        Quando o {time} for mencionado, você deve:
        1. Informações sobre Calendário e Competições

        Pesquise e liste os próximos 3-5 jogos do time, incluindo:

        Data e horário
        Adversário
        Competição
        Local da partida (casa ou fora)

        2. Situação Atual nas Competições

        Identifique todas as competições que o time está disputando atualmente
        Para cada competição, forneça:

        Posição atual na tabela
        Pontuação
        Aproveitamento (%)
        Distância para o líder ou para a zona de classificação/rebaixamento
        Chances matemáticas (quando disponíveis)

        3. Notícias Relevantes

        Busque as 10 notícias mais relevantes e recentes sobre o time
        Priorize notícias com base em:

        Atualidade (últimos 7 dias)
        Relevância para o momento do clube
        Repercussão entre torcedores (engajamento)
        Fontes confiáveis de informação esportiva

        4. Análise de Tendências

        Analise o sentimento geral das notícias (positivo, neutro ou negativo)
        Identifique temas recorrentes (ex: desempenho de jogadores específicos, questões de gestão, etc.)
        Observe o nível de entusiasmo ou preocupação da torcida com base nas notícias

        Formato de Apresentação
        Estruture sua resposta da seguinte forma:
        1. RESUMO DA SITUAÇÃO ATUAL
        Um parágrafo introdutório resumindo o momento atual do time, destacando os aspectos mais relevantes encontrados na pesquisa.
        2. PRÓXIMOS COMPROMISSOS
        Lista formatada dos próximos jogos do time.
        3. CLASSIFICAÇÃO NAS COMPETIÇÕES
        Informações organizadas sobre a posição do time em cada competição que disputa.
        4. PRINCIPAIS NOTÍCIAS
        Apresente as 10 notícias mais relevantes em formato de parágrafo, cada uma contendo:

        Título destacado da notícia
        Breve resumo do conteúdo (2-3 frases)
        Data de publicação
        Link direto para a fonte original
        Separação clara entre as diferentes notícias

        5. ANÁLISE DE TENDÊNCIAS
        Um parágrafo final analisando as tendências observadas nas notícias e o momento do clube.
        Observações Importantes

        Utilize fontes confiáveis de informação esportiva
        Verifique a data das notícias para garantir que são recentes
        Forneça links diretos para todas as fontes utilizadas
        Apresente informações de forma neutra e objetiva
        Certifique-se de que todas as informações são precisas e atualizadas
        Se houver informações limitadas ou contraditórias, mencione esta situação

        Este formato garante que o usuário receba um panorama completo e atualizado sobre o time solicitado, com informações organizadas de        maneira clara e acessível.
        """,
        description="Agente que busca notcias atuais do time no Google",
        tools=[google_search]
    )

    entrada_do_agente_buscador = f"Time: {time}\nData de hoje: {data_de_hoje}"

    noticias = call_agent(buscador, entrada_do_agente_buscador)
    return noticias


def agente_torcedor(time):
    buscador = Agent(
        name="agente_torcedor",
        model="gemini-2.0-flash",
        instruction=f"""
        Você é um torcedor fanático que também é poeta. Ao falar sobre o {time} de futebol específico use informacoes atuais por meio da                  tool do google searche e você deve:
        Demonstre conhecimento técnico sobre futebol
        Expresse paixão intensa pelo time mencionado
        Use linguagem emotiva e metafórica
        Mantenha lealdade inabalável, mesmo em momentos difíceis

        Estrutura da Resposta
        1. Análise do Momento Atual (2 parágrafos)

        Avalie o desempenho recente do time nas competições
        Comente sobre jogadores importantes, técnico e estilo de jogo
        Mencione a situação nas tabelas e próximos desafios
        Use metáforas e comparações para descrever o momento esportivo

        2. Conexão Emocional (1 parágrafo)

        Expresse como o momento atual do time afeta suas emoções
        Relacione a situação do clube com sentimentos profundos

        3. Poema Original (8-12 linhas)
        Crie um poema que:

        Tenha título relacionado ao momento do time, use dados de jogadores atuais conforme pesquisa no google search
        Use as cores e símbolos do clube como elementos poéticos
        Capture a essência do momento atual (glória, superação, reconstrução
        Inclua metáforas originais que conectem futebol e emoções
        Termine com um verso marcante sobre sua devoção ao clube

        4. Conclusão Breve (1 frase)

        Reafirme seu amor incondicional pelo time, independente das circunstâncias

        Orientações Gerais

        Adapte o tom ao momento real do time (entusiasmado ou reflexivo)
        Seja específico sobre o time mencionado
        Balance crítica honesta com admiração apaixonada
        Evite clichês futebolísticos e poéticos comuns

        O objetivo é combinar uma análise lúcida do momento atual do time com expressão poética autêntica que demonstre sua paixão profunda pelo         clube.
        """,
        description="Agente que busca notcias atuais do time no Google",
        tools=[google_search]
    )

    entrada_do_agente_buscador = f"Time: {time}"

    visao_torcedor = call_agent(buscador, entrada_do_agente_buscador)
    return visao_torcedor
    # Carrega o modelo de NLP para português (carrega apenas uma vez)
@st.cache_resource
def carregar_modelo_nlp():
    try:
        # Carrega o modelo em português
        return spacy.load('pt_core_news_lg')
    except OSError:
        # Se o modelo não estiver instalado, tenta instalar
        st.warning("Instalando modelo de linguagem português para spaCy. Isso pode demorar um pouco...")
        from spacy.cli import download
        download('pt_core_news_lg')
        return spacy.load('pt_core_news_lg')

# Função para extrair nomes de pessoas do texto
def extrair_nomes_pessoas(texto):
    """
    Extrai nomes de pessoas (jogadores, técnicos, etc.) do texto usando NLP.
    
    Args:
        texto (str): Texto para analisar
        
    Returns:
        dict: Dicionário com nomes e suas contagens
    """
    # Carrega o modelo
    nlp = carregar_modelo_nlp()
    
    # Processa o texto
    doc = nlp(texto)
    
    # Extrai entidades do tipo pessoa (PESSOA, PER)
    nomes_pessoas = []
    
    for ent in doc.ents:
        # Em pt_core_news_lg, pessoas são marcadas como 'PER'
        if ent.label_ == 'PER':
            # Normaliza o nome (remove espaços extras, converte para minúsculas)
            nome = ent.text.strip()
            # Adiciona o nome encontrado à lista
            nomes_pessoas.append(nome)
    
    # Conta a frequência de cada nome
    contador = Counter(nomes_pessoas)
    
    # Converte para dicionário (formato esperado pelo WordCloud)
    return dict(contador)

# Função para extrair texto das notícias da resposta da LLM
def extrair_texto_noticias(resposta_texto):
    # Esta função extrai o texto das notícias da resposta da LLM
    # Podemos supor que as notícias estão em formato de lista ou após algum marcador
    
    # Tenta identificar a seção de notícias (ajuste conforme o formato da resposta da LLM)
    noticias_texto = ""
    
    # Padrões para identificar seções de notícias na resposta
    padroes = [
        r"Principais notícias.*?:(.*?)(?=Próximo jogo|$)",
        r"Notícias recentes.*?:(.*?)(?=Próximo jogo|$)",
        r"Notícias dos últimos.*?dias:(.*?)(?=Próximo jogo|$)",
        r"Últimas notícias:(.*?)(?=Próximo jogo|$)"
    ]
    
    # Tenta cada padrão
    for padrao in padroes:
        match = re.search(padrao, resposta_texto, re.DOTALL | re.IGNORECASE)
        if match:
            noticias_texto = match.group(1)
            break
    
    # Se não encontrou com os padrões específicos, pega todo o texto
    if not noticias_texto:
        noticias_texto = resposta_texto
    
    # Remove URLs e caracteres especiais
    noticias_texto = re.sub(r'https?://\S+', '', noticias_texto)
    
    return noticias_texto

# Função para gerar nuvem de palavras
def gerar_nuvem_palavras(texto,largura=800, altura=400, 
                         palavras_para_excluir=None,
                         apenas_pessoas=False):
    """
    Gera uma nuvem de palavras a partir do texto fornecido.
    
    Args:
        texto (str): Texto para gerar a nuvem de palavras
        largura (int): Largura da imagem
        altura (int): Altura da imagem
        palavras_para_excluir (list): Lista de palavras a serem excluídas
        apenas_pessoas (bool): Se True, apenas nomes de pessoas serão usados
        
    Returns:
        fig: Figura matplotlib com a nuvem de palavras
    """
    if apenas_pessoas:
        # Extrai nomes de pessoas do texto
        frequencias = extrair_nomes_pessoas(texto)
        
        # Se não encontrou nenhum nome
        if not frequencias:
            return None
        
        # Cria a nuvem de palavras com as frequências fornecidas
        wordcloud = WordCloud(
            width=largura,
            height=altura,
            background_color='white',
            min_font_size=10,
            max_font_size=200,
            colormap='viridis',
        ).generate_from_frequencies(frequencias)
    else:
        # Lista padrão de stopwords em português
        stopwords_pt = [
            "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo", "as", "até",
            "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles", "depois",
            "do", "dos", "e", "é", "ela", "elas", "ele", "eles", "em", "entre", "era",
            "eram", "éramos", "essa", "essas", "esse", "esses", "esta", "estas", "este",
            "estes", "eu", "foi", "fomos", "for", "foram", "fosse", "fossem", "fui", "há",
            "isso", "isto", "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu",
            "meus", "minha", "minhas", "muito", "na", "não", "nas", "nem", "no", "nos",
            "nós", "nossa", "nossas", "nosso", "nossos", "num", "numa", "o", "os", "ou",
            "para", "pela", "pelas", "pelo", "pelos", "por", "qual", "quando", "que",
            "quem", "são", "se", "seja", "sejam", "sem", "será", "seu", "seus", "só", "sobre",
            "somos", "sua", "suas", "também", "te", "tem", "tém", "temos", "ter", "teu",
            "teus", "tu", "tua", "tuas", "um", "uma", "umas", "uns", "vos", "você", "vocês"
        ]
        
        # Adiciona palavras personalizadas para excluir
        if palavras_para_excluir:
            stopwords_pt.extend(palavras_para_excluir)
        
        # Cria a nuvem de palavras
        wordcloud = WordCloud(
            width=largura,
            height=altura,
            background_color='white',
            stopwords=set(stopwords_pt),
            min_font_size=10,
            max_font_size=200,
            colormap='viridis',
            collocations=False
        ).generate(texto)
    
    # Cria uma figura matplotlib
    fig, ax = plt.subplots(figsize=(largura/100, altura/100))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    
    return fig

# Configuração da página
st.set_page_config(
    page_title="Brasileirão 2025",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Função para carregar a chave de API de forma segura
def get_api_key():
    try:
        if 'GOOGLE_API_KEY' in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception as e:
        st.warning(f"Não foi possível carregar secrets: {e}")
    
    # Fallback para quando os secrets não estão disponíveis
    return st.text_input("Insira sua API Key do Google:", type="password")
    
    # Opção 2: Usar variáveis de ambiente (bom para desenvolvimento local)
    #if 'GOOGLE_API_KEY' in os.environ:
    #    return os.environ['GOOGLE_API_KEY'] 


# CSS para remover elementos da interface padrão do Streamlit e customizar os botões
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Estilo para o novo layout de time */
    .time-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 15px;
    }
    
    .time-img {
        width: 60px;
        height: 60px;
        object-fit: contain;
        margin-bottom: 5px;
    }
    
    /* Centralizar imagens e botões */
    div[data-testid="column"] {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }
    
    div[data-testid="stImage"] {
        display: flex;
        justify-content: center;
        margin-bottom: 5px;
    }
    
    /* Remover margem padrão dos containers */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0;
    }
    </style>
""", unsafe_allow_html=True)

# Lista dos times do Brasileirão 2025
times = [
    {"nome": "Flamengo", "logo": "https://logodetimes.com/times/flamengo/logo-flamengo-256.png"},
    {"nome": "Palmeiras", "logo": "https://logodetimes.com/times/palmeiras/logo-palmeiras-256.png"},
    {"nome": "São Paulo", "logo": "https://logodetimes.com/times/sao-paulo/logo-sao-paulo-256.png"},
    {"nome": "Corinthians", "logo": "https://logodetimes.com/times/corinthians/logo-corinthians-256.png"},
    {"nome": "Fluminense", "logo": "https://logodetimes.com/times/fluminense/logo-fluminense-256.png"},
    {"nome": "Botafogo", "logo": "https://logodetimes.com/times/botafogo/logo-botafogo-256.png"},
    {"nome": "Vasco", "logo": "https://logodetimes.com/times/vasco-da-gama/logo-vasco-da-gama-256.png"},
    {"nome": "Grêmio", "logo": "https://logodetimes.com/times/gremio/logo-gremio-256.png"},
    {"nome": "Internacional", "logo": "https://logodetimes.com/times/internacional/logo-internacional-256.png"},
    {"nome": "Cruzeiro", "logo": "https://logodetimes.com/times/cruzeiro/logo-cruzeiro-256.png"},
    {"nome": "Atlético-MG", "logo": "https://logodetimes.com/times/atletico-mineiro/logo-atletico-mineiro-256.png"},
    {"nome": "Mirassol", "logo": "https://logodetimes.com/times/mirassol/logo-mirassol-256.png"},
    {"nome": "Bahia", "logo": "https://logodetimes.com/times/bahia/logo-bahia-256.png"},
    {"nome": "Santos", "logo": "https://logodetimes.com/times/santos/logo-santos-256.png"},
    {"nome": "Fortaleza", "logo": "https://logodetimes.com/times/fortaleza/logo-fortaleza-256.png"},
    {"nome": "Bragantino", "logo": "https://logodetimes.com/times/red-bull-bragantino/logo-red-bull-bragantino-256.png"},
    {"nome": "Vitória", "logo": "https://logodetimes.com/times/vitoria/logo-vitoria-256.png"},
    {"nome": "Ceará", "logo": "https://logodetimes.com/times/ceara/logo-ceara-256.png"},
    {"nome": "Juventude", "logo": "https://logodetimes.com/times/juventude/logo-juventude-256.png"},
    {"nome": "Sport", "logo": "https://logodetimes.com/times/sport-recife/logo-sport-recife-256.png"}
]

# Função para criar um fallback de imagem quando a URL não estiver disponível
def get_image_placeholder():
    # Cria uma imagem em branco com um texto "Time"
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', (100, 100), color=(240, 240, 240))
    d = ImageDraw.Draw(img)
    # Adiciona um texto simples
    d.text((30, 40), "Time", fill=(0, 0, 0))
    
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Função para carregar imagem de URL com fallback
def load_image_from_url(url):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except:
        return get_image_placeholder()

# Função para consultar informações do time usando Gemini
def consultar_time_info(time_nome, agente_nome_selecionado):
    api_key = get_api_key()    
    if not api_key:
        st.warning("⚠️ Chave de API não configurada. Configure a chave para usar os recursos de IA.")
        return None
    try:
        data_de_hoje = date.today().strftime("%d/%m/%Y")
        resposta_agente = None # Inicializa

        if agente_nome_selecionado == "Noticias_atuais":
            resposta_agente = agente_noticias(time_nome, data_de_hoje)
        elif agente_nome_selecionado == "História_Titulos":
            resposta_agente = agente_historia(time_nome)
        elif agente_nome_selecionado == "Analista_Esportivo":
            resposta_agente = agente_analista(time_nome, agente_noticias(time_nome, data_de_hoje))
        elif agente_nome_selecionado == "Torcedor_Poeta":
            resposta_agente = agente_torcedor(time_nome)
        else:
            st.error(f"Agente '{agente_nome_selecionado}' desconhecido.")
            return None # Falha

        return resposta_agente

    except Exception as e:
        st.error(f"Erro ao processar a solicitação com o agente '{agente_nome_selecionado}': {str(e)}")
        return None # Falha


# Título da aplicação
st.markdown("<h1 style='text-align: center; color: blue;'>BrasileirIA: O Futebol Nacional sob Múltiplos Olhares</h1>", unsafe_allow_html=True)

# --- INÍCIO DA ADIÇÃO DO SELECTBOX PARA AGENTE ---
nomes_agentes_disponiveis = [
    "Noticias_atuais",
    "História_Titulos",
    "Analista_Esportivo",
    "Torcedor_Poeta"
]

# Inicializar o agente selecionado no session_state se não existir
if 'agente_selecionado_usuario' not in st.session_state:
    st.session_state.agente_selecionado_usuario = nomes_agentes_disponiveis[0]  # Padrão é "Agente de Notícias"

# Widget de seleção do agente
agente_escolhido_pelo_usuario = st.selectbox(
    "Selecione o Agente de IA:",
    options=nomes_agentes_disponiveis,
    index=nomes_agentes_disponiveis.index(st.session_state.agente_selecionado_usuario), # Mantém a seleção
    key="selectbox_agente_principal"
)
# Atualiza o estado da sessão com a escolha do usuário
if st.session_state.agente_selecionado_usuario != agente_escolhido_pelo_usuario:
    st.session_state.agente_selecionado_usuario = agente_escolhido_pelo_usuario
    # Se o agente mudar e um time já estiver selecionado, podemos querer limpar o resultado anterior
    # ou forçar um rerun para buscar com o novo agente. Por ora, apenas atualiza o estado.
    # Se um time estiver selecionado, a próxima ação (como voltar) ou um rerun forçado aqui
    # poderia reprocessar. Por simplicidade, deixaremos a atualização e o fluxo normal.
# --- FIM DA ADIÇÃO DO SELECTBOX PARA AGENTE ---

st.divider()

# Inicializar session_state para time selecionado se não existir
if 'time_selecionado' not in st.session_state:
    st.session_state.time_selecionado = None

# Se nenhum time estiver selecionado, mostra a grade de seleção
if not st.session_state.time_selecionado:
    # Organiza os times em uma grid de 4 linhas e 5 colunas
    # Total de 20 times (4 x 5)
    linhas = 4
    colunas = 5
    
    for linha in range(linhas):
        # Criar as colunas para cada linha
        cols = st.columns(colunas)
        
        for coluna in range(colunas):
            # Índice do time atual
            idx = linha * colunas + coluna
            
            # Verifica se ainda há times para mostrar
            if idx < len(times):
                time = times[idx]
                
                with cols[coluna]:
                    # Carregar imagem do time
                    encoded_image = load_image_from_url(time["logo"])
                    
                    # Container centralizado com CSS inline mais forte
                    st.markdown(f'''
                    <div style="display: flex; flex-direction: column; align-items: center; width: 100%; margin-bottom: 10px;">
                        <img src="data:image/png;base64,{encoded_image}" width="60" style="margin: 0 auto; display: block;">
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Botão com o nome do time
                    if st.button(time["nome"], key=f"btn_{time['nome']}", use_container_width=True):
                        st.session_state.time_selecionado = time["nome"]
                        st.rerun()
else:
    # Mostra informações do time selecionado
    time_selecionado = st.session_state.time_selecionado
    
    # Encontra o time selecionado na lista para mostrar sua imagem
    time_info = next((t for t in times if t["nome"] == time_selecionado), None)
    
    if time_info:
        col1, col2 = st.columns([1, 3])        
        with col1:
            encoded_image = load_image_from_url(time_info["logo"])
            st.image(f"data:image/png;base64,{encoded_image}", width=100)
        
        with col2:
            st.subheader(f"Conteúdo gerado pelo Agente {st.session_state.agente_selecionado_usuario}")
    else:
        st.subheader(f"Informações sobre o {time_selecionado}")
    
    # Container para mostrar um spinner durante o carregamento
    with st.spinner(f"Buscando informações sobre o {time_selecionado}..."):
        # Consulta informações do time usando Gemini
        info_time = consultar_time_info(time_selecionado, st.session_state.agente_selecionado_usuario)
        
        if info_time:
            # Exibe as informações do time
            st.markdown(info_time)
            
            # Extrai o texto das notícias da resposta da LLM
            texto_noticias = extrair_texto_noticias(info_time)
            
            # Cria abas para diferentes tipos de nuvens de palavras
            tab1, tab2 = st.tabs(["Nuvem de Termos Gerais", "Nuvem de Pessoas"])
            
            with tab1:
                # Adiciona palavras específicas para excluir da nuvem
                palavras_para_excluir = [time_selecionado, "time", "jogo", "jogador", "jogadores", "clube", "dia"]
                
                # Gera a nuvem de palavras normal
                if texto_noticias.strip():
                    fig = gerar_nuvem_palavras(
                        texto_noticias, 
                        palavras_para_excluir=palavras_para_excluir
                    )
                    
                    # Exibe a nuvem de palavras
                    st.pyplot(fig)
                else:
                    st.info("Não foram encontradas notícias suficientes para gerar a nuvem de palavras.")
            
            with tab2:
                with st.spinner("Processando nomes de pessoas..."):
                    try:
                        # Tenta gerar a nuvem de pessoas
                        fig_pessoas = gerar_nuvem_palavras(
                            texto_noticias,
                            apenas_pessoas=True
                        )
                        
                        if fig_pessoas:
                            st.pyplot(fig_pessoas)
                        else:
                            st.info("Não foram encontrados nomes de pessoas suficientes nas notícias.")
                    except Exception as e:
                        st.error(f"Erro ao processar nomes de pessoas: {str(e)}")
                        st.info("Dica: Você precisa instalar o pacote spaCy e o modelo de linguagem português com os comandos:")
                        st.code("pip install spacy\npython -m spacy download pt_core_news_lg")
        else:
            st.error("Não foi possível obter informações. Verifique sua chave API.")
    
    # Botão para voltar à seleção de times
    if st.button("← Voltar para seleção de times", use_container_width=True):
        st.session_state.time_selecionado = None
        st.rerun()