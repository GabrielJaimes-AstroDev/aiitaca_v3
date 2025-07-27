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
        background-color: #15181c !important;  /* Color plomo oscuro */
    }
    
    /* Texto general en blanco/tonos claros */
    body, .stMarkdown, .stText, 
    .stSlider [data-testid="stMarkdownContainer"],
    .stSelectbox, .stNumberInput, .stTextInput,
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF !important;
    }
    
    /* Sidebar blanco con texto oscuro (sin cambios) */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    
    /* Botones azules (mejor contraste con fondo oscuro) */
    .stButton>button {
        border: 2px solid #1E88E5 !important;
        color: white !important;
        background-color: #1E88E5 !important;
        border-radius: 5px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s !important;
    }
    .stButton>button:hover {
        border: 2px solid #0D47A1 !important;
        background-color: #0D47A1 !important;
    }
    
    /* Títulos y encabezados */
    h1, h2, h3, h4, h5, h6 {
        color: #1E88E5 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    h1 {
        border-bottom: 2px solid #1E88E5 !important;
        padding-bottom: 10px !important;
    }
    
    /* Panel de descripción adaptado */
    .description-panel {
        text-align: justify;
        background-color: white !important;
        color: black !important;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #1E88E5 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .description-panel p {
        margin-bottom: 15px;
        line-height: 1.6;
        color: #000000 !important;
    }
    .description-panel strong {
        color: #000000 !important;
        font-weight: bold;
    }
    
    /* Sliders y controles */
    .stSlider .thumb {
        background-color: #1E88E5 !important;
    }
    .stSlider .track {
        background-color: #5F9EA0 !important;  /* Tonos que combinan con plomo */
    }
    
    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        color: #FFFFFF !important;
        background-color: #81acde !important;  /* Tono intermedio */
        border-radius: 5px 5px 0 0;
        border: 1px solid #1E88E5;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    
    /* File uploader adaptado */
    .stFileUploader>div>div>div>div {
        background-color: #E5E7E9 !important;  /* Plomo muy claro */
        border: 2px solid #B0E0E6 !important;
        color: #000000 !important;
        border-radius: 5px !important;
        padding: 10px !important;
    }
    .stFileUploader>div>div>div>div:hover {
        background-color: #D5D8DC !important;  /* Un poco más oscuro al hover */
        border-color: #1E88E5 !important;
    }
    .stFileUploader>div>section>div>div>div>span {
        color: #000000 !important;
        font-size: 14px !important;
    }
    .stFileUploader>div>section>div>button {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    .stFileUploader>div>section>div>button:hover {
        background-color: #0D47A1 !important;
    }
    
    /* Ajustes para gráficos y visualizaciones */
    .stPlotlyChart, .stDataFrame {
        background-color: #1e88e5 !important;
        border-radius: 8px;
        padding: 10px;
    }
    
    /* Mejoras para inputs y selects */
    .stTextInput input, .stNumberInput input, 
    .stSelectbox select {
        background-color: #1e88e5 !important;
        color: white !important;
        border: 1px solid #5F9EA0 !important;
    }
    
    /* Ocultar títulos de contenido de pestañas */
    .tab-content h2, .tab-content h3, .tab-content h4 {
        display: none !important;
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
    
    /* Fondo del gráfico interactivo */
    .js-plotly-plot .plotly, .plot-container {
        background-color: #0D0F14 !important;
    }
    
    /* Configuración del gráfico Plotly */
    .plotly-graph-div {
        background-color: #0D0F14 !important;
    }
    
    /* Nuevos estilos para paneles de información */
    .info-panel {
        background-color: white !important;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: black !important;
    }
    .info-panel h3 {
        color: #1E88E5 !important;
        margin-top: 0;
    }
    .info-panel img {
        max-width: 100%;
        height: auto;
        border-radius: 5px;
        margin: 10px 0;
    }
    .pro-tip {
        background-color: #f0f7ff !important;
        padding: 12px;
        border-radius: 5px;
        border-left: 4px solid #1E88E5;
        margin-top: 15px;
    }
    .pro-tip p {
        margin: 0;
        font-size: 0.9em;
        color: #333 !important;
    }
    
    /* Estilo para la barra de progreso */
    .progress-bar {
        margin-top: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# === HEADER WITH IMAGE AND DESCRIPTION ===
st.image("NGC6523_BVO_2.jpg", use_container_width=True)

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

# === CONFIGURATION ===
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1zlnkEoRvHR1CoK9hXxD0Jy4JIKF5Uybz?usp=drive_link"
TEMP_MODEL_DIR = "downloaded_models"

if not os.path.exists(TEMP_MODEL_DIR):
    os.makedirs(TEMP_MODEL_DIR)

@st.cache_data(show_spinner=True)
@st.cache_data(show_spinner=True)
def download_models_from_drive(folder_url, output_dir):
    model_files = [f for f in os.listdir(output_dir) if f.endswith('.keras')]
    data_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]

    if model_files and data_files:
        return model_files, data_files, True

    try:
        progress_text = st.sidebar.empty()
        progress_bar = st.sidebar.progress(0)
        progress_text.text("📥 Preparing to download models...")
        
        # Primero verificamos cuántos archivos hay que descargar
        file_count = 0
        try:
            file_count = 10  # Valor estimado para la simulación
        except:
            file_count = 10  # Valor por defecto si no podemos obtener el conteo real
            
        with st.spinner("📥 Downloading models from Google Drive..."):
            gdown.download_folder(
                folder_url, 
                output=output_dir, 
                quiet=True,  # Silenciamos la salida por consola
                use_cookies=False
            )
            for i in range(file_count):
                time.sleep(0.5)  # Pequeña pausa para simular descarga
                progress = int((i + 1) / file_count * 100)
                progress_bar.progress(progress)
                progress_text.text(f"📥 Downloading models... {progress}%")
        
        model_files = [f for f in os.listdir(output_dir) if f.endswith('.keras')]
        data_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]
        
        progress_bar.progress(100)
        progress_text.text("Process Completed")
        
        if model_files and data_files:
            st.sidebar.success("✅ Models downloaded successfully!")
        else:
            st.sidebar.error("❌ No models found in the specified folder")
            
        return model_files, data_files, True
    except Exception as e:
        st.sidebar.error(f"❌ Error downloading models: {str(e)}")
        return [], [], False

# === SIDEBAR ===
st.sidebar.title("Configuration")

model_files, data_files, models_downloaded = download_models_from_drive(GDRIVE_FOLDER_URL, TEMP_MODEL_DIR)

input_file = st.sidebar.file_uploader(
    "Input Spectrum File ( . | .txt | .dat | .fits | .spec )",
    type=None,
    help="Drag and drop file here ( . | .txt | .dat | .fits | .spec ). Limit 200MB per file"
)

st.sidebar.subheader("Peak Matching Parameters")
sigma_emission = st.sidebar.slider("Sigma Emission", 0.1, 5.0, 1.5, step=0.1)
window_size = st.sidebar.slider("Window Size", 1, 20, 3, step=1)
sigma_threshold = st.sidebar.slider("Sigma Threshold", 0.1, 5.0, 2.0, step=0.1)
fwhm_ghz = st.sidebar.slider("FWHM (GHz)", 0.01, 0.5, 0.05, step=0.01)
tolerance_ghz = st.sidebar.slider("Tolerance (GHz)", 0.01, 0.5, 0.1, step=0.01)
min_peak_height_ratio = st.sidebar.slider("Min Peak Height Ratio", 0.1, 1.0, 0.3, step=0.05)
top_n_lines = st.sidebar.slider("Top N Lines", 5, 100, 30, step=5)
top_n_similar = st.sidebar.slider("Top N Similar", 50, 2000, 800, step=50)

config = {
    'trained_models_dir': TEMP_MODEL_DIR,
    'peak_matching': {
        'sigma_emission': sigma_emission,
        'window_size': window_size,
        'sigma_threshold': sigma_threshold,
        'fwhm_ghz': fwhm_ghz,
        'tolerance_ghz': tolerance_ghz,
        'min_peak_height_ratio': min_peak_height_ratio,
        'top_n_lines': top_n_lines,
        'debug': True,
        'top_n_similar': top_n_similar
    }
}

# === MAIN APP ===
st.title("Molecular Spectrum Analyzer | AI - ITACA")

if input_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(input_file.getvalue())
        tmp_path = tmp_file.name

    if not model_files:
        st.error("No trained models were found in Google Drive.")
    else:
        selected_model = st.selectbox("Select Molecule Model", model_files)
        
        # Contenedor para el botón y las pestañas de información
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            analyze_btn = st.button("Analyze Spectrum")
        
        with col2:
            params_tab = st.button("📝 Parameters Explanation", key="params_btn", 
                                  help="Click to show parameters explanation")
        
        with col3:
            flow_tab = st.button("📊 Flow of Work Diagram", key="flow_btn", 
                               help="Click to show the workflow diagram")
        
        if params_tab:
            with st.container():
                st.markdown("""
                <div class="description-panel">
                    <h3 style="text-align: center; margin-top: 0; color: black; border-bottom: 2px solid #1E88E5; padding-bottom: 10px;">Technical Parameters Guide</h3>
                    
                <div style="margin-bottom: 25px;">
                <h4 style="color: #1E88E5; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 15px;">🔬 Peak Detection</h4>
                <p><strong>Sigma Emission (1.5):</strong> Threshold for peak detection in standard deviations (σ) of the noise. 
                <span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Higher values reduce false positives but may miss weak peaks. Typical range: 1.0-3.0</span></p>
                
                <p><strong>Window Size (3):</strong> Points in Savitzky-Golay smoothing kernel. 
                <span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Odd integers only. Larger values smooth noise but blur close peaks.</span></p>
                
                <p><strong>Sigma Threshold (2.0):</strong> Minimum peak prominence (σ). 
                <span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Filters low-significance features. Critical for crowded spectra.</span></p>
                
                <p><strong>FWHM (0.05 GHz):</strong> Expected line width at half maximum. 
                <span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Should match your instrument's resolution. Affects line fitting.</span></p>
                </div>
                    
                <div style="margin-bottom: 25px;">
                <h4 style="color: #1E88E5; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 15px;">🔄 Matching Parameters</h4>
                <p><strong>Tolerance (0.1 GHz):</strong> Frequency matching window. 
                <span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Accounts for Doppler shifts (±20 km/s at 100 GHz). Increase for broad lines.</span></p>
                
                <p><strong>Min Peak Ratio (0.3):</strong> Relative intensity cutoff. 
                <span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Peaks below this fraction of strongest line are excluded. Range: 0.1-0.5.</span></p>
                </div>
                
                <div>
                <h4 style="color: #1E88E5; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 15px;">📊 Output Settings</h4>
                <p><strong>Top N Lines (30):</strong> Lines displayed in results. 
                <span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Doesn't affect analysis quality, only visualization density.</span></p>
                
                <p><strong>Top N Similar (800):</strong> Synthetic spectra retained. 
                <span style="display: block; margin-left: 20px; font-size: 0.92em; color: #555;">Higher values improve accuracy but increase runtime. Max: 2000.</span></p>
                </div>
                
                <div style="margin-top: 20px; padding: 12px; background-color: #f8f9fa; border-radius: 5px; border-left: 4px solid #1E88E5;">
                <p style="margin: 0; font-size: 0.9em;"><strong>Pro Tip:</strong> For ALMA data (high resolution), start with FWHM=0.05 GHz and Tolerance=0.05 GHz. For single-dish telescopes, try FWHM=0.2 GHz.</p>
                </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Mostrar el diagrama de flujo si se hace clic
        if flow_tab:
            with st.container():
                st.markdown("""
                    <div class="info-panel">
                        <h3 style="text-align: center; color: black; border-bottom: 2px solid #1E88E5; padding-bottom: 10px;">Flow of Work Diagram</h3>
                    </div>
                """, unsafe_allow_html=True)

                st.image("Flow_of_Work.jpg", use_container_width=True)

                st.markdown("""
                    <div style="margin-top: 20px;">
                    <h4 style="color: #1E88E5; margin-bottom: 10px;">Analysis Pipeline Steps:</h4>
                    <ol style="color: white; padding-left: 20px;">
                    <li><strong>Spectrum Input:</strong> Upload your observational spectrum</li>
                    <li><strong>Pre-processing:</strong> Noise reduction and baseline correction</li>
                    <li><strong>Peak Detection:</strong> Identify significant spectral features</li>
                    <li><strong>Model Matching:</strong> Compare with synthetic spectra database</li>
                    <li><strong>Parameter Estimation:</strong> Determine physical conditions (T<sub>ex</sub>, logN)</li>
                    <li><strong>Visualization:</strong> Interactive comparison of observed vs synthetic spectra</li>
                    </ol>
                    </div>
                    <div class="pro-tip">
                    <p><strong>Note:</strong> The complete analysis typically takes 30-90 seconds depending on spectrum complexity and selected parameters.</p>
                    </div>
                """, unsafe_allow_html=True)


        if analyze_btn:
            try:
                # Configurar la barra de progreso para el análisis
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                def update_analysis_progress(step, total_steps=6):
                    progress = int((step / total_steps) * 100)
                    progress_bar.progress(progress)
                    steps = [
                        "Loading model...",
                        "Processing spectrum...",
                        "Detecting peaks...",
                        "Matching with database...",
                        "Calculating parameters...",
                        "Generating visualizations..."
                    ]
                    progress_text.text(f"🔍 Analyzing spectrum... {steps[step-1]} ({progress}%)")
                
                update_analysis_progress(1)
                mol_name = selected_model.replace('_model.keras', '')

                model_path = os.path.join(TEMP_MODEL_DIR, selected_model)
                model = tf.keras.models.load_model(model_path)

                update_analysis_progress(2)
                data_file = os.path.join(TEMP_MODEL_DIR, f'{mol_name}_train_data.npz')
                if not os.path.exists(data_file):
                    st.error(f"Training data not found for {mol_name}")
                else:
                    with np.load(data_file) as data:
                        train_freq = data['train_freq']
                        train_data = data['train_data']
                        train_logn = data['train_logn']
                        train_tex = data['train_tex']
                        headers = data['headers']
                        filenames = data['filenames']

                    update_analysis_progress(3)
                    results = analyze_spectrum(
                        tmp_path, model, train_data, train_freq,
                        filenames, headers, train_logn, train_tex,
                        config, mol_name
                    )

                    update_analysis_progress(6)
                    st.success("Analysis completed successfully!")

                    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "Interactive Summary", 
                        "Molecule Best Match", 
                        "Peak Matching", 
                        "CNN Training", 
                        "Top Selection: LogN", 
                        "Top Selection: Tex"
                    ])

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
                            
                            st.plotly_chart(fig, use_container_width=True)

                    with tab1:
                        if results['best_match']:
                            st.pyplot(plot_summary_comparison(
                                results['input_freq'], results['input_spec'],
                                results['best_match'], tmp_path
                            ))

                    with tab2:
                        if results['best_match']:
                            st.pyplot(plot_zoomed_peaks_comparison(
                                results['input_spec'], results['input_freq'],
                                results['best_match']
                            ))

                    with tab3:
                        st.pyplot(plot_best_matches(
                            results['train_logn'], results['train_tex'],
                            results['similarities'], results['distances'],
                            results['closest_idx_sim'], results['closest_idx_dist'],
                            results['train_filenames'], results['input_logn']
                        ))

                    with tab4:
                        st.pyplot(plot_tex_metrics(
                            results['train_tex'], results['train_logn'],
                            results['similarities'], results['distances'],
                            results['top_similar_indices'],
                            results['input_tex'], results['input_logn']
                        ))

                    with tab5:
                        st.pyplot(plot_similarity_metrics(
                            results['train_logn'], results['train_tex'],
                            results['similarities'], results['distances'],
                            results['top_similar_indices'],
                            results['input_logn'], results['input_tex']
                        ))

                    # Limpiar la barra de progreso al finalizar
                    progress_bar.empty()
                    progress_text.empty()

                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

else:
    st.info("Please upload an input spectrum file to begin analysis.")

# Instructions
st.sidebar.markdown("""
**Instructions:**
1. Select the directory containing the trained models
2. Upload your input spectrum file ( . | .txt | .dat | .fits | .spec )
3. Adjust the peak matching parameters as needed
4. Select the model to use for analysis
5. Click 'Analyze Spectrum' to run the analysis

**Interactive Plot Controls:**
- 🔍 Zoom: Click and drag to select area
- 🖱️ Hover: View exact values
- 🔄 Reset: Double-click
- 🏎️ Pan: Shift+click+drag
- 📊 Range Buttons: Quick zoom to percentage ranges
""")