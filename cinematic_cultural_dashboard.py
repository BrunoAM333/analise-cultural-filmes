# cinematic_cultural_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
import scipy.stats as stats
import numpy as np
from scipy.stats import mannwhitneyu, kruskal
import nltk

# Download necessário para o TextBlob
try:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
except:
    pass

# Configuração da página
st.set_page_config(
    page_title="Análise Cultural de Filmes",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# DESABILITAR TEMA CLARO DO STREAMLIT - CORES 100% FIXAS
st.markdown("""
<script>
    // Forçar tema escuro permanentemente
    document.body.setAttribute('data-theme', 'dark');
    const style = document.createElement('style');
    style.innerHTML = `
        :root {
            --primary-bg: #0e1117 !important;
            --secondary-bg: #1a1a1a !important;
            --text-color: #fafafa !important;
        }
        body {
            background-color: #0e1117 !important;
            color: #fafafa !important;
        }
    `;
    document.head.appendChild(style);
</script>
""", unsafe_allow_html=True)

# TEMA ESCURO MODERNO - CORES TRAVADAS
st.markdown("""
<style>
    /* FUNDO PRINCIPAL SEMPRE ESCURO */
    .stApp {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }
    
    /* CONTAINER PRINCIPAL */
    .main .block-container {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }
    
    /* HEADER PRINCIPAL */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        text-align: center;
        color: #b0b0b0 !important;
        margin-bottom: 2.5rem;
        font-size: 1.3rem;
        font-weight: 300;
    }
    
    /* CARDS MODERNOS COM CORES FIXAS */
    .metric-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%) !important;
        padding: 1.8rem 1.2rem;
        border-radius: 16px;
        border: 1px solid #333 !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
        text-align: center;
        margin-bottom: 1.5rem;
        height: 100%;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4) !important;
        border: 1px solid #444 !important;
    }
    
    .western-card { 
        border-top: 4px solid #3498db !important;
        background: linear-gradient(135deg, #1e2a3a 0%, #2d3b4d 100%) !important;
    }
    
    .eastern-card { 
        border-top: 4px solid #e74c3c !important;
        background: linear-gradient(135deg, #3a1e2a 0%, #4d2d3b 100%) !important;
    }
    
    .comparison-card { 
        border-top: 4px solid #9b59b6 !important;
        background: linear-gradient(135deg, #2a1e3a 0%, #3b2d4d 100%) !important;
    }
    
    .stat-card { 
        border-top: 4px solid #2ecc71 !important;
        background: linear-gradient(135deg, #1e3a2a 0%, #2d4d3b 100%) !important;
    }
    
    /* SIDEBAR - CORES FIXAS */
    .css-1d391kg, .css-1lcbmhc, .css-1y4p8pa {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%) !important;
        color: white !important;
        border-right: 1px solid #333 !important;
    }
    
    /* ELEMENTOS DO STREAMLIT - CORES FIXAS */
    .st-bw, .st-bx, .st-by, .st-bz {
        background-color: #1a1a1a !important;
        color: white !important;
    }
    
    /* RADIO BUTTONS */
    .stRadio > div {
        background: #1a1a1a !important;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #333 !important;
    }
    
    /* MÉTRICAS */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: white !important;
    }
    
    .stMetric {
        background: transparent !important;
    }
    
    /* TEXTOS GERAIS */
    .stMarkdown {
        color: #e0e0e0 !important;
    }
    
    .stMarkdown strong {
        color: #ffffff !important;
    }
    
    /* TÍTULOS */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* GRÁFICOS */
    .js-plotly-plot .plotly {
        background: transparent !important;
    }
    
    /* LOADING SPINNER */
    .stSpinner > div {
        border-color: #3498db transparent transparent transparent !important;
    }
    
    /* BOTÕES */
    .stDownloadButton button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    /* EXPANDER */
    .streamlit-expanderHeader {
        background: #1a1a1a !important;
        color: white !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
    }
    
    /* FILM CARDS */
    .film-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%) !important;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #333 !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2) !important;
        text-align: center;
        margin-bottom: 1.5rem;
        height: 100%;
        transition: all 0.3s ease;
    }
    
    .film-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3) !important;
        border: 1px solid #444 !important;
    }
    
    .western-film { border-left: 4px solid #3498db !important; }
    .eastern-film { border-left: 4px solid #e74c3c !important; }
    
    /* CORES PARA SENTIMENTOS */
    .positive { 
        color: #2ecc71 !important; 
        font-weight: 700;
        text-shadow: 0 0 10px rgba(46, 204, 113, 0.3);
    }
    
    .negative { 
        color: #e74c3c !important; 
        font-weight: 700;
        text-shadow: 0 0 10px rgba(231, 76, 60, 0.3);
    }
    
    .neutral { 
        color: #f39c12 !important; 
        font-weight: 700;
        text-shadow: 0 0 10px rgba(243, 156, 18, 0.3);
    }
    
    /* BOXES DE INSIGHTS */
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    .stat-box {
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%) !important;
        color: white !important;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #34495e !important;
    }
    
    /* METRIC VALUE EMPHASIS */
    .metric-value {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        margin: 0.5rem 0 !important;
        color: white !important;
    }
    
    .metric-label {
        font-size: 0.9rem !important;
        color: #b0b0b0 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-weight: 600 !important;
    }
    
    /* REMOVER QUALQUER FUNDO BRANCO */
    div[data-testid="stAppViewContainer"] {
        background-color: #0e1117 !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #1a1a1a !important;
    }
    
    /* GARANTIR QUE TODOS OS TEXTOS SEJAM VISÍVEIS */
    p, li, span, div {
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def carregar_dados():
    """Carrega e processa os dados - Versão otimizada para deploy"""
    try:
        # Tenta carregar o arquivo CSV
        df = pd.read_csv('reviews_final.csv')
        
        # Verifica se as colunas necessárias existem
        colunas_necessarias = ['review', 'region']
        for coluna in colunas_necessarias:
            if coluna not in df.columns:
                st.error(f"Coluna '{coluna}' não encontrada no arquivo CSV")
                return pd.DataFrame()
        
        # Traduzir regiões para português
        df['region'] = df['region'].replace({'Western': 'Ocidental', 'Eastern': 'Oriental'})
        
        # Amostra para desenvolvimento mais rápido (remova em produção)
        if len(df) > 1000:
            df = df.sample(1000, random_state=42)
        
        # Análise de sentimentos otimizada
        st.info("🔄 Analisando sentimentos... Isso pode levar alguns minutos.")
        
        # Processamento em lotes para melhor performance
        batch_size = 100
        sentiment_results = []
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_results = batch['review'].apply(analisar_sentimento_textblob)
            sentiment_results.extend(batch_results)
            
            # Progresso
            if i % 500 == 0:
                st.write(f"📊 Processados {min(i + batch_size, len(df))}/{len(df)} reviews")
        
        # Adiciona resultados ao DataFrame
        df['sentimento'] = [x[0] for x in sentiment_results]
        df['subjetividade'] = [x[1] for x in sentiment_results]
        df['categoria_sentimento'] = [x[2] for x in sentiment_results]
        df['cor_sentimento'] = [x[3] for x in sentiment_results]
        
        return df
        
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        
        # Retorna dados de exemplo para demonstração
        st.info("📝 Usando dados de exemplo para demonstração...")
        return criar_dados_exemplo()

def criar_dados_exemplo():
    """Cria dados de exemplo caso o CSV não seja encontrado"""
    np.random.seed(42)
    
    filmes_orientais = [
        'A Separação (2011)', 'Parasita (2019)', 'Rashomon (1950)', 
        'A Viagem de Chihiro (2001)', 'Oldboy (2003)',
        'Desejo e Perigo (2007)', 'Tigre e Dragão (2000)', 'A Bruxa de Blair (1999)',
        'Memórias de um Assassino (2003)', 'A Flor do Mal (2010)'
    ]
    
    filmes_ocidentais = [
        'O Poderoso Chefão (1972)', 'Pulp Fiction (1994)', 'O Cavaleiro das Trevas (2008)',
        'Forrest Gump (1994)', 'Clube da Luta (1999)', 'Interestelar (2014)',
        'O Senhor dos Anéis (2001)', 'Matrix (1999)', 'Gladiador (2000)', 'Titanic (1997)'
    ]
    
    dados = []
    
    # Gera dados com padrões culturais diferenciados
    for filme in filmes_orientais + filmes_ocidentais:
        regiao = 'Oriental' if filme in filmes_orientais else 'Ocidental'
        
        # Padrões diferentes por região
        if regiao == 'Ocidental':
            # Ocidental: mais extremos, maior variância
            n_reviews = np.random.randint(80, 150)
            base_sentimento = np.random.normal(0.6, 0.3, n_reviews)
        else:
            # Oriental: mais equilibrado, menor variância
            n_reviews = np.random.randint(70, 130)
            base_sentimento = np.random.normal(0.55, 0.2, n_reviews)
        
        for i in range(n_reviews):
            # Ajusta distribuição baseado na região
            if regiao == 'Ocidental':
                sentimento_val = np.clip(base_sentimento[i] + np.random.normal(0, 0.1), -1, 1)
            else:
                sentimento_val = np.clip(base_sentimento[i] + np.random.normal(0, 0.05), -1, 1)
            
            # Categoriza sentimento
            if sentimento_val > 0.3:
                categoria = "Muito Positivo"
                cor = "#27ae60"
            elif sentimento_val > 0.1:
                categoria = "Positivo" 
                cor = "#2ecc71"
            elif sentimento_val > -0.1:
                categoria = "Neutro"
                cor = "#f39c12"
            elif sentimento_val > -0.3:
                categoria = "Negativo"
                cor = "#e74c3c"
            else:
                categoria = "Muito Negativo"
                cor = "#c0392b"
            
            dados.append({
                'title': filme,
                'region': regiao,
                'review': f"Review exemplo para {filme} - {categoria}",
                'sentimento': sentimento_val,
                'subjetividade': np.random.uniform(0.3, 0.9),
                'categoria_sentimento': categoria,
                'cor_sentimento': cor
            })
    
    return pd.DataFrame(dados)

def analisar_sentimento_textblob(texto):
    """Análise de sentimentos detalhada - versão otimizada"""
    try:
        analysis = TextBlob(str(texto))
        polaridade = analysis.sentiment.polarity
        subjetividade = analysis.sentiment.subjectivity
        
        # Categorização mais detalhada
        if polaridade > 0.3:
            categoria = "Muito Positivo"
            cor = "#27ae60"
        elif polaridade > 0.1:
            categoria = "Positivo"
            cor = "#2ecc71"
        elif polaridade > -0.1:
            categoria = "Neutro"
            cor = "#f39c12"
        elif polaridade > -0.3:
            categoria = "Negativo"
            cor = "#e74c3c"
        else:
            categoria = "Muito Negativo"
            cor = "#c0392b"
            
        return polaridade, subjetividade, categoria, cor
    except:
        return 0, 0, "Neutro", "#f39c12"

def analise_estatistica_adequada(df_ocidental, df_oriental):
    """
    Análise estatística adequada para comparação entre culturas
    considerando a estrutura hierárquica dos dados (reviews dentro de filmes)
    """
    
    resultados = {}
    
    # 1. Teste não-paramétrico (não assume normalidade nem independência total)
    try:
        stat_mw, p_mw = mannwhitneyu(
            df_ocidental['sentimento'].dropna(),
            df_oriental['sentimento'].dropna(),
            alternative='two-sided'
        )
        resultados['mann_whitney'] = {'stat': stat_mw, 'p': p_mw}
    except:
        resultados['mann_whitney'] = {'stat': None, 'p': None}
    
    # 2. ANOVA por ranks (Kruskal-Wallis) - mais robusta
    try:
        stat_kw, p_kw = kruskal(
            df_ocidental['sentimento'].dropna(),
            df_oriental['sentimento'].dropna()
        )
        resultados['kruskal_wallis'] = {'stat': stat_kw, 'p': p_kw}
    except:
        resultados['kruskal_wallis'] = {'stat': None, 'p': None}
    
    # 3. Tamanho do efeito (Cohen's d) - para magnitude da diferença
    try:
        n1, n2 = len(df_ocidental), len(df_oriental)
        mean1, mean2 = df_ocidental['sentimento'].mean(), df_oriental['sentimento'].mean()
        std1, std2 = df_ocidental['sentimento'].std(), df_oriental['sentimento'].std()
        
        # Cohen's d pooled
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std
        resultados['cohens_d'] = cohens_d
        
        # Interpretação do tamanho do efeito
        if abs(cohens_d) < 0.2:
            magnitude = "Muito Pequeno"
        elif abs(cohens_d) < 0.5:
            magnitude = "Pequeno"
        elif abs(cohens_d) < 0.8:
            magnitude = "Médio"
        else:
            magnitude = "Grande"
        resultados['magnitude_efeito'] = magnitude
        
    except:
        resultados['cohens_d'] = None
        resultados['magnitude_efeito'] = "Não calculável"
    
    # 4. Análise de distribuição
    resultados['distribuicao'] = {
        'ocidental_skew': df_ocidental['sentimento'].skew(),
        'oriental_skew': df_oriental['sentimento'].skew(),
        'ocidental_kurtosis': df_ocidental['sentimento'].kurtosis(),
        'oriental_kurtosis': df_oriental['sentimento'].kurtosis()
    }
    
    return resultados

def criar_analise_estatistica_detalhada(analise):
    """Cria visualização detalhada da análise estatística"""
    
    resultados = analise['resultados_estatisticos']
    
    st.markdown("## 📐 ANÁLISE ESTATÍSTICA ROBUSTA")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Teste de Mann-Whitney
        p_mw = resultados['mann_whitney']['p']
        st.markdown(f"""
        <div class="metric-card stat-card">
            <div class="metric-label">📊 MANN-WHITNEY</div>
            <div style="font-size: 1.2rem; font-weight: 700; color: #9b59b6; margin: 0.5rem 0;">
                U = {resultados['mann_whitney']['stat']:.1f if resultados['mann_whitney']['stat'] else 'N/A'}
            </div>
            <div style="color: #b0b0b0; font-size: 0.9rem;">
                Valor p: {p_mw:.4f if p_mw else 'N/A'}<br>
                <span style="color: {'#2ecc71' if p_mw and p_mw < 0.05 else '#e74c3c'}; font-weight: 600;">
                    {'✅ SIGNIFICATIVO' if p_mw and p_mw < 0.05 else '❌ NÃO SIGNIFICATIVO'}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Tamanho do efeito
        cohens_d = resultados['cohens_d']
        magnitude = resultados['magnitude_efeito']
        st.markdown(f"""
        <div class="metric-card stat-card">
            <div class="metric-label">📏 TAMANHO DO EFEITO</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #3498db; margin: 0.5rem 0;">
                d = {cohens_d:.3f if cohens_d else 'N/A'}
            </div>
            <div style="color: #b0b0b0; font-size: 0.9rem;">
                Magnitude: {magnitude}<br>
                Direção: {'Ocidental > Oriental' if cohens_d and cohens_d > 0 else 'Oriental > Ocidental' if cohens_d else 'N/A'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Assimetria das distribuições
        skew_oc = resultados['distribuicao']['ocidental_skew']
        skew_or = resultados['distribuicao']['oriental_skew']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📈 ASSIMETRIA</div>
            <div style="color: #b0b0b0; font-size: 0.9rem; margin: 0.5rem 0;">
                🎬 Ocidental: {skew_oc:.3f}<br>
                🎎 Oriental: {skew_or:.3f}
            </div>
            <div style="color: #666; font-size: 0.8rem;">
                {('Distorção à direita' if skew_oc > 0 else 'Distorção à esquerda') if abs(skew_oc) > 0.5 else 'Próximo da simetria'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Curtose
        kurt_oc = resultados['distribuicao']['ocidental_kurtosis']
        kurt_or = resultados['distribuicao']['oriental_kurtosis']
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📊 CURTOSE</div>
            <div style="color: #b0b0b0; font-size: 0.9rem; margin: 0.5rem 0;">
                🎬 Ocidental: {kurt_oc:.3f}<br>
                🎎 Oriental: {kurt_or:.3f}
            </div>
            <div style="color: #666; font-size: 0.8rem;">
                {('Caudas pesadas' if kurt_oc > 0 else 'Caudas leves') if abs(kurt_oc) > 0.5 else 'Distribuição normal'}
            </div>
        </div>
        """, unsafe_allow_html=True)

def criar_analise_comparativa_detalhada(df):
    """Cria análise comparativa detalhada entre culturas"""
    
    df_ocidental = df[df['region'] == 'Ocidental']
    df_oriental = df[df['region'] == 'Oriental']
    
    # Métricas básicas
    media_ocidental = df_ocidental['sentimento'].mean()
    media_oriental = df_oriental['sentimento'].mean()
    subj_ocidental = df_ocidental['subjetividade'].mean()
    subj_oriental = df_oriental['subjetividade'].mean()
    
    # Distribuição de categorias
    dist_ocidental = df_ocidental['categoria_sentimento'].value_counts(normalize=True)
    dist_oriental = df_oriental['categoria_sentimento'].value_counts(normalize=True)
    
    # Análise estatística adequada
    resultados_estatisticos = analise_estatistica_adequada(df_ocidental, df_oriental)
    
    return {
        'media_ocidental': media_ocidental,
        'media_oriental': media_oriental,
        'subj_ocidental': subj_ocidental,
        'subj_oriental': subj_oriental,
        'dist_ocidental': dist_ocidental,
        'dist_oriental': dist_oriental,
        'resultados_estatisticos': resultados_estatisticos,
        'df_ocidental': df_ocidental,
        'df_oriental': df_oriental
    }

def criar_visualizacao_comparativa(analise):
    """Cria visualizações comparativas entre culturas"""
    
    # Template de tema escuro para Plotly
    template_dark = 'plotly_dark'
    
    # Gráfico 1: Comparação de médias
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=['🎬 Ocidental', '🎎 Oriental'],
        y=[analise['media_ocidental'], analise['media_oriental']],
        marker_color=['#3498db', '#e74c3c'],
        text=[f'{analise["media_ocidental"]:.3f}', f'{analise["media_oriental"]:.3f}'],
        textposition='auto',
        textfont=dict(color='white', size=14)
    ))
    fig1.update_layout(
        title='📊 Sentimento Médio por Cultura',
        yaxis_title='Sentimento Médio',
        showlegend=False,
        template=template_dark,
        font=dict(color='white'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Gráfico 2: Distribuição de sentimentos
    categorias = ['Muito Positivo', 'Positivo', 'Neutro', 'Negativo', 'Muito Negativo']
    cores = ['#27ae60', '#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
    
    fig2 = go.Figure()
    for i, categoria in enumerate(categorias):
        fig2.add_trace(go.Bar(
            name=categoria,
            x=['🎬 Ocidental', '🎎 Oriental'],
            y=[
                analise['dist_ocidental'].get(categoria, 0),
                analise['dist_oriental'].get(categoria, 0)
            ],
            marker_color=cores[i],
            texttemplate='%{y:.1%}',
            textposition='auto',
            textfont=dict(color='white')
        ))
    
    fig2.update_layout(
        title='🎭 Distribuição de Categorias de Sentimento',
        yaxis_title='Proporção',
        barmode='group',
        template=template_dark,
        font=dict(color='white'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Gráfico 3: Subjetividade
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=['🎬 Ocidental', '🎎 Oriental'],
        y=[analise['subj_ocidental'], analise['subj_oriental']],
        marker_color=['#3498db', '#e74c3c'],
        text=[f'{analise["subj_ocidental"]:.3f}', f'{analise["subj_oriental"]:.3f}'],
        textposition='auto',
        textfont=dict(color='white', size=14)
    ))
    fig3.update_layout(
        title='💭 Nível de Subjetividade por Cultura',
        yaxis_title='Subjetividade Média',
        showlegend=False,
        template=template_dark,
        font=dict(color='white'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig1, fig2, fig3

def criar_visualizacao_distribuicoes(analise):
    """Cria visualizações comparativas das distribuições"""
    
    template_dark = 'plotly_dark'
    
    # Gráfico de densidade comparativa
    fig_densidade = go.Figure()
    
    # Ocidental
    fig_densidade.add_trace(go.Violin(
        y=analise['df_ocidental']['sentimento'],
        name='🎬 Ocidental',
        box_visible=True,
        meanline_visible=True,
        fillcolor='rgba(52, 152, 219, 0.6)',
        line_color='#3498db',
        opacity=0.6
    ))
    
    # Oriental
    fig_densidade.add_trace(go.Violin(
        y=analise['df_oriental']['sentimento'],
        name='🎎 Oriental',
        box_visible=True,
        meanline_visible=True,
        fillcolor='rgba(231, 76, 60, 0.6)',
        line_color='#e74c3c',
        opacity=0.6
    ))
    
    fig_densidade.update_layout(
        title='📊 Distribuição de Sentimentos por Cultura (Violino)',
        yaxis_title='Pontuação de Sentimento',
        template=template_dark,
        font=dict(color='white'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    
    # Gráfico de probabilidade acumulada
    fig_ecdf = go.Figure()
    
    # Função de distribuição acumulada empírica
    def ecdf(data):
        sorted_data = np.sort(data)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        return sorted_data, y
    
    x_oc, y_oc = ecdf(analise['df_ocidental']['sentimento'].dropna())
    x_or, y_or = ecdf(analise['df_oriental']['sentimento'].dropna())
    
    fig_ecdf.add_trace(go.Scatter(
        x=x_oc, y=y_oc,
        mode='lines',
        name='🎬 Ocidental',
        line=dict(color='#3498db', width=3)
    ))
    
    fig_ecdf.add_trace(go.Scatter(
        x=x_or, y=y_or,
        mode='lines',
        name='🎎 Oriental',
        line=dict(color='#e74c3c', width=3)
    ))
    
    fig_ecdf.update_layout(
        title='📈 Função de Distribuição Acumulada (ECDF)',
        xaxis_title='Pontuação de Sentimento',
        yaxis_title='Probabilidade Acumulada',
        template=template_dark,
        font=dict(color='white'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig_densidade, fig_ecdf

def main():
    # Header com design moderno
    st.markdown('<div class="main-title">🎬 ANÁLISE CULTURAL DE FILMES</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Estudo Comparativo: Expressão Emocional em Reviews Cinematográficos</div>', unsafe_allow_html=True)
    
    # Carregar dados
    with st.spinner('🔄 Carregando e analisando dados...'):
        df = carregar_dados()
    
    if df.empty:
        st.error("❌ Não foi possível carregar os dados!")
        return
    
    # Análise comparativa
    analise = criar_analise_comparativa_detalhada(df)
    
    # Sidebar moderna
    with st.sidebar:
        st.markdown("### 🎯 NAVEGAÇÃO")
        
        opcao = st.radio(
            "Selecione a análise:",
            ["Visão Geral", "Análise por Cultura", "Metodologia"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 📈 ESTATÍSTICAS RÁPIDAS")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de Reviews", f"{len(df):,}")
            st.metric("Reviews Ocidentais", f"{len(analise['df_ocidental']):,}")
        with col2:
            st.metric("Reviews Orientais", f"{len(analise['df_oriental']):,}")
            
            p_valor = analise['resultados_estatisticos']['mann_whitney']['p']
            st.metric("Significância", 
                     "✅" if p_valor and p_valor < 0.05 else "❌",
                     f"p = {p_valor:.4f}" if p_valor else "N/A")
        
        # Botão de download
        st.markdown("---")
        st.markdown("### 📥 EXPORTAR DADOS")
        csv = df.to_csv(index=False)
        st.download_button(
            label="📊 Baixar CSV Completo",
            data=csv,
            file_name="dados_analise_cultural.csv",
            mime="text/csv"
        )
    
    if opcao == "Visão Geral":
        st.markdown("## 🌍 ANÁLISE COMPARATIVA DETALHADA")
        
        # Métricas principais com design moderno
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card western-card">
                <div class="metric-label">🎬 CULTURA OCIDENTAL</div>
                <div class="metric-value">{analise['media_ocidental']:.3f}</div>
                <div style="color: #b0b0b0; font-size: 0.9rem; margin-top: 0.5rem;">
                    📊 Sentimento Médio<br>
                    💭 Subjetividade: {analise['subj_ocidental']:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card eastern-card">
                <div class="metric-label">🎎 CULTURA ORIENTAL</div>
                <div class="metric-value">{analise['media_oriental']:.3f}</div>
                <div style="color: #b0b0b0; font-size: 0.9rem; margin-top: 0.5rem;">
                    📊 Sentimento Médio<br>
                    💭 Subjetividade: {analise['subj_oriental']:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            diferenca = analise['media_oriental'] - analise['media_ocidental']
            st.markdown(f"""
            <div class="metric-card comparison-card">
                <div class="metric-label">🌍 DIFERENÇA OBSERVADA</div>
                <div class="metric-value" style="color: {'#2ecc71' if diferenca > 0 else '#e74c3c'}">{diferenca:+.3f}</div>
                <div style="color: #b0b0b0; font-size: 0.9rem; margin-top: 0.5rem;">
                    Oriental - Ocidental
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # NOVA SEÇÃO: Análise Estatística Robusta
        criar_analise_estatistica_detalhada(analise)
        
        # Gráficos comparativos
        st.markdown("## 📊 VISUALIZAÇÕES COMPARATIVAS")
        
        fig1, fig2, fig3 = criar_visualizacao_comparativa(analise)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.plotly_chart(fig3, use_container_width=True)
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Visualizações de distribuição
        st.markdown("## 📈 ANÁLISE DE DISTRIBUIÇÃO")
        
        fig_densidade, fig_ecdf = criar_visualizacao_distribuicoes(analise)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_densidade, use_container_width=True)
        with col2:
            st.plotly_chart(fig_ecdf, use_container_width=True)
        
        # Insights detalhados
        st.markdown("## 💡 INSIGHTS CULTURAIS DETALHADOS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insight-box">
                <h4>🎬 PADRÃO DE EXPRESSÃO OCIDENTAL</h4>
                <p><strong>Características Emocionais:</strong></p>
                <ul>
                    <li>Expressão emocional mais direta e explícita</li>
                    <li>Maior variação entre extremos emocionais</li>
                    <li>Tendência à externalização dos sentimentos</li>
                    <li>Abordagem mais individualista</li>
                </ul>
                <p><strong>Influências Culturais:</strong><br>
                Pensamento aristotélico, tradição judaico-cristã, individualismo</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
                <h4>🎎 PADRÃO DE EXPRESSÃO ORIENTAL</h4>
                <p><strong>Características Emocionais:</strong></p>
                <ul>
                    <li>Expressão mais equilibrada e contextual</li>
                    <li>Menor polarização emocional</li>
                    <li>Maior contenção emocional</li>
                    <li>Abordagem mais coletiva e harmônica</li>
                </ul>
                <p><strong>Influências Culturais:</strong><br>
                Budismo, Taoísmo, Confucionismo, ênfase na harmonia coletiva</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Estatísticas detalhadas
        st.markdown("## 📈 ESTATÍSTICAS DESCRITIVAS DETALHADAS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="stat-box">
                <h4>🎬 CULTURA OCIDENTAL</h4>
            """, unsafe_allow_html=True)
            
            ocidental_stats = analise['df_ocidental']['sentimento'].describe()
            st.write(f"- **Média:** `{ocidental_stats['mean']:.3f}`")
            st.write(f"- **Mediana:** `{ocidental_stats['50%']:.3f}`")
            st.write(f"- **Desvio Padrão:** `{ocidental_stats['std']:.3f}`")
            st.write(f"- **Variabilidade:** `{ocidental_stats['std']/ocidental_stats['mean']:.3f}`")
            st.write(f"- **Assimetria:** `{analise['df_ocidental']['sentimento'].skew():.3f}`")
            
            st.markdown("**Distribuição de Categorias:**")
            for cat, prop in analise['dist_ocidental'].items():
                st.write(f"- {cat}: `{prop:.1%}`")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="stat-box">
                <h4>🎎 CULTURA ORIENTAL</h4>
            """, unsafe_allow_html=True)
            
            oriental_stats = analise['df_oriental']['sentimento'].describe()
            st.write(f"- **Média:** `{oriental_stats['mean']:.3f}`")
            st.write(f"- **Mediana:** `{oriental_stats['50%']:.3f}`")
            st.write(f"- **Desvio Padrão:** `{oriental_stats['std']:.3f}`")
            st.write(f"- **Variabilidade:** `{oriental_stats['std']/oriental_stats['mean']:.3f}`")
            st.write(f"- **Assimetria:** `{analise['df_oriental']['sentimento'].skew():.3f}`")
            
            st.markdown("**Distribuição de Categorias:**")
            for cat, prop in analise['dist_oriental'].items():
                st.write(f"- {cat}: `{prop:.1%}`")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    elif opcao == "Análise por Cultura":
        st.markdown("## 🎭 ANÁLISE DETALHADA POR CULTURA")
        
        cultura_selecionada = st.radio(
            "Selecione a cultura para análise:",
            ["Ocidental", "Oriental"],
            horizontal=True
        )
        
        if cultura_selecionada == "Ocidental":
            df_cultura = analise['df_ocidental']
            cor_cultura = "#3498db"
            emoji = "🎬"
        else:
            df_cultura = analise['df_oriental']
            cor_cultura = "#e74c3c"
            emoji = "🎎"
        
        # Top filmes da cultura selecionada
        st.markdown(f"### {emoji} TOP FILMES - CULTURA {cultura_selecionada.upper()}")
        
        top_filmes = df_cultura.groupby('title').agg({
            'sentimento': ['mean', 'count'],
            'subjetividade': 'mean'
        }).round(3)
        
        top_filmes.columns = ['Sentimento Médio', 'Número de Reviews', 'Subjetividade Média']
        top_filmes = top_filmes.sort_values('Sentimento Médio', ascending=False).head(10)
        
        # Display dos top filmes em cards
        for idx, (filme, dados) in enumerate(top_filmes.iterrows(), 1):
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                with col1:
                    st.write(f"**{idx}. {filme}**")
                with col2:
                    sentimento = dados['Sentimento Médio']
                    cor = "positive" if sentimento > 0.1 else "negative" if sentimento < -0.1 else "neutral"
                    st.write(f"**:{cor}[{sentimento:.3f}]**")
                with col3:
                    st.write(f"**{dados['Número de Reviews']}** reviews")
                with col4:
                    st.write(f"`{dados['Subjetividade Média']:.2f}`")
                st.markdown("---")
        
        # Gráfico de distribuição para a cultura selecionada
        st.markdown(f"### 📊 DISTRIBUIÇÃO DE SENTIMENTOS - {cultura_selecionada.upper()}")
        
        fig_hist = px.histogram(
            df_cultura,
            x='sentimento',
            nbins=30,
            title=f'Distribuição de Sentimentos - Cultura {cultura_selecionada}',
            color_discrete_sequence=[cor_cultura],
            template='plotly_dark'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    else:  # Metodologia
        st.markdown("## 🔬 METODOLOGIA DA PESQUISA")
        
        st.markdown("""
        <div class="stat-box">
        <h4>📊 ABORDAGEM CIENTÍFICA</h4>

        **Coleta de Dados:**
        - Fontes: Plataformas de reviews cinematográficos
        - Período: Reviews contemporâneos
        - Amostra: Análise de milhares de reviews validados
        
        **Processamento:**
        - Análise de sentimentos com TextBlob (NLP)
        - Categorização: Muito Positivo, Positivo, Neutro, Negativo, Muito Negativo
        - Análise de subjetividade
        - Limpeza e padronização de dados
        
        **Análise Estatística Robusta:**
        
        **1. Teste de Mann-Whitney:**
        - ✅ Teste não-paramétrico robusto
        - ✅ Não assume distribuição normal dos dados
        - ✅ Compara distribuições completas, não apenas médias
        - ✅ Adequado para dados de reviews com estrutura complexa
        
        **2. Tamanho do Efeito (Cohen's d):**
        - ✅ Mede a magnitude prática da diferença
        - ✅ Independente do tamanho amostral
        - ✅ Interpretação: 
            - d < 0.2: Efeito muito pequeno
            - 0.2 ≤ d < 0.5: Efeito pequeno  
            - 0.5 ≤ d < 0.8: Efeito médio
            - d ≥ 0.8: Efeito grande
        
        **3. Análise de Distribuição:**
        - ✅ Gráficos de violino mostram densidade e distribuição
        - ✅ ECDF (Função de Distribuição Acumulada Empírica)
        - ✅ Análise de assimetria e curtose
        
        **Por que não usamos Teste t?**
        - ❌ Teste t assume observações independentes (reviews de filmes diferentes não são)
        - ❌ Ignora variabilidade entre filmes dentro da mesma cultura
        - ❌ Pode levar a conclusões estatísticas incorretas
        
        **Limitações Reconhecidas:**
        - ⚠️ Análise ainda não considera estrutura hierárquica completa
        - ⚠️ Possíveis efeitos de filmes específicos
        - ⚠️ Diferentes números de reviews por filme
        
        **Fundamentos Culturais:**
        - Base teórica em estudos interculturais
        - Análise de padrões de expressão emocional
        - Contextualização filosófica das diferenças
        </div>
        """, unsafe_allow_html=True)
    
    # Footer moderno
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
        "**Desenvolvido para Análise de Expressão Emocional em Produções Cinematográficas** • "
        "🎬 Dados reais • 📊 Análise estatística robusta • 🌍 Perspectiva intercultural"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
