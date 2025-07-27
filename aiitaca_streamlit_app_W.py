import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from core_functions import *
import tempfile
import plotly.graph_objects as go
import tensorflow as tf
import gdown
import shutil
import time

st.set_page_config(
    layout="wide", 
    page_title="AI-ITACA | Spectrum Analyzer",
    page_icon="🔭" 
)

# === CUSTOM CSS STYLES ===
st.markdown("""
<style>
    /* Fondo principal color plomo oscuro y texto claro */
    .stApp, .main .block-container, body {
        background-color: #15181c !important;
    }
    
    /* Texto general en blanco/tonos claros */
    body, .stMarkdown, .stText, 
    .stSlider [data-testid="stMarkdownContainer"],
    .stSelectbox, .stNumberInput, .stTextInput,
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF !important;
    }
    
    /* Sidebar blanco con texto oscuro */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    /* Botones azules */
    .stButton>button {
        border: 2px solid #1E88E5 !important;
        color: white !important;
        background-color: #1E88E5 !important;
        border-radius: 5px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s !important;
    }
    
    /* Títulos y encabezados */
    h1, h2, h3, h4, h5, h6 {
        color: #1E88E5 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Panel de descripción */
    .description-panel {
        text-align: justify;
        background-color: white !important;
        color: black !important;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #1E88E5 !important;
    }
    
    /* Configuración del gráfico Plotly */
    .plotly-graph-div {
        background-color: #0D0F14 !important;
    }
    
    /* Estilo para los valores de Physical Parameters */
    .physical-params {
        color: #000000 !important;
        font-size: 1.1rem !important;
        margin: 5px 0 !important;
    }
    
    /* Panel azul claro para el resumen */
    .summary-panel {
        background-color: #FFFFFF !important;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# === HEADER WITH IMAGE AND DESCRIPTION ===
st.image("NGC6523_BVO_2.jpg", use_column_width=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.empty()
    
with col2:
    st.markdown("""
    <style>
        .main-title {
            color: white !important;
            font-size: 2.5rem !important;
            font-weight: bold !important;
        }
        .subtitle {
            color: white !important;
            font-size: 1.5rem !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-title">AI-ITACA | Artificial Intelligence Integral Tool for AstroChemical Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Molecular Spectrum Analyzer</p>', unsafe_allow_html=True)

# Project description
st.markdown("""
<div class="description-panel" style="text-align: justify;">
A remarkable upsurge in the complexity of molecules identified in the interstellar medium (ISM) is currently occurring, with over 80 new species discovered in the last three years. A number of them have been emphasized by prebiotic experiments as vital molecular building blocks of life. Since our Solar System was formed from a molecular cloud in the ISM, it prompts the query as to whether the rich interstellar chemical reservoir could have played a role in the emergence of life. The improved sensitivities of state-of-the-art astronomical facilities, such as the Atacama Large Millimeter/submillimeter Array (ALMA) and the James Webb Space Telescope (JWST), are revolutionizing the discovery of new molecules in space. However, we are still just scraping the tip of the iceberg. We are far from knowing the complete catalogue of molecules that astrochemistry can offer, as well as the complexity they can reach.<br><br>
<strong>Artificial Intelligence Integral Tool for AstroChemical Analysis (AI-ITACA)</strong>, proposes to combine complementary machine learning (ML) techniques to address all the challenges that astrochemistry is currently facing. AI-ITACA will significantly contribute to the development of new AI-based cutting-edge analysis software that will allow us to make a crucial leap in the characterization of the level of chemical complexity in the ISM, and in our understanding of the contribution that interstellar chemistry might have in the origin of life.
</div>
""", unsafe_allow_html=True)

# === SIDEBAR ===
st.sidebar.title("Configuration")

# File uploader in sidebar
input_file = st.sidebar.file_uploader(
    "Input Spectrum File ( . | .txt | .dat | .fits | .spec )",
    type=None,
    help="Drag and drop file here ( . | .txt | .dat | .fits | .spec ). Limit 200MB per file"
)

# === MAIN APP ===
if input_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(input_file.getvalue())
        tmp_path = tmp_file.name

    # Simulación de resultados para la pestaña interactiva
    results = {
        'input_freq': np.linspace(100, 120, 1000),
        'input_spec': np.random.normal(0, 1, 1000) + np.sin(np.linspace(0, 10, 1000)),
        'best_match': {
            'x_synth': np.linspace(100, 120, 1000),
            'y_synth': np.random.normal(0, 0.8, 1000) + np.sin(np.linspace(0, 10, 1000)),
            'logn': 15.2,
            'tex': 45.5,
            'filename': "simulated_match.txt"
        }
    }

    # Interactive Summary Tab
    tab0 = st.tabs(["Interactive Summary"])[0]
    
    with tab0:
        if results.get('best_match'):
            st.markdown(f"""
            <div class="summary-panel">
                <h4 style="color: #1E88E5; margin-top: 0;">Detection of Physical Parameters</h4>
                <p class="physical-params"><strong>LogN:</strong> {results['best_match']['logn']:.2f} cm⁻²</p>
                <p class="physical-params"><strong>Tex:</strong> {results['best_match']['tex']:.2f} K</p>
                <p class="physical-params"><strong>File (Top CNN Train):</strong> {results['best_match']['filename']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=results['input_freq'],
                y=results['input_spec'],
                mode='lines',
                name='Input Spectrum',
                line=dict(color='white', width=2))
            )
            
            fig.add_trace(go.Scatter(
                x=results['best_match']['x_synth'],
                y=results['best_match']['y_synth'],
                mode='lines',
                name='Best Match',
                line=dict(color='red', width=2))
            )
            
            fig.update_layout(
                plot_bgcolor='#0D0F14',
                paper_bgcolor='#0D0F14',
                margin=dict(l=50, r=50, t=60, b=50),
                xaxis_title='Frequency (GHz)',
                yaxis_title='Intensity (K)',
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=600,
                font=dict(color='white'),
                xaxis=dict(gridcolor='#3A3A3A'),
                yaxis=dict(gridcolor='#3A3A3A')
            )
            
            st.plotly_chart(fig, use_column_width=True)

    os.unlink(tmp_path)
else:
    st.info("Please upload an input spectrum file to begin analysis.")

# Instructions in sidebar
st.sidebar.markdown("""
**Instructions:**
1. Upload your input spectrum file ( . | .txt | .dat | .fits | .spec )
2. The analysis will begin automatically
3. View results in the Interactive Summary tab

**Interactive Plot Controls:**
- 🔍 Zoom: Click and drag to select area
- 🖱️ Hover: View exact values
- 🔄 Reset: Double-click
""")
