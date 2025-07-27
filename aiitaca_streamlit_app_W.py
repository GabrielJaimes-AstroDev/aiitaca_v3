import streamlit as st
import os
import numpy as np
import tempfile
import plotly.graph_objects as go
import gdown
import time
from scipy.interpolate import interp1d
from astropy.io import fits
import zipfile
import shutil

# Configuración de la página
st.set_page_config(
    layout="wide", 
    page_title="AI-ITACA | Spectrum Analyzer",
    page_icon="🔭" 
)

# === ESTILOS CSS ===
st.markdown("""
<style>
    .stApp, .main .block-container, body {
        background-color: #15181c !important;
    }
    body, .stMarkdown, .stText, 
    .stSlider [data-testid="stMarkdownContainer"],
    .stSelectbox, .stNumberInput, .stTextInput,
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }
    .stButton>button {
        border: 2px solid #1E88E5 !important;
        color: white !important;
        background-color: #1E88E5 !important;
        border-radius: 5px !important;
        padding: 0.5rem 1rem !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1E88E5 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .description-panel {
        text-align: justify;
        background-color: white !important;
        color: black !important;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 5px solid #1E88E5 !important;
    }
    .plotly-graph-div {
        background-color: #0D0F14 !important;
    }
    .physical-params {
        color: #000000 !important;
        font-size: 1.1rem !important;
        margin: 5px 0 !important;
    }
    .summary-panel {
        background-color: #FFFFFF !important;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 20px;
    }
    .filter-tab {
        background-color: #1e1e1e !important;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# === ENCABEZADO ===
st.image("NGC6523_BVO_2.jpg", use_column_width=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.empty()
    
with col2:
    st.markdown('<p class="main-title">AI-ITACA | Artificial Intelligence Integral Tool for AstroChemical Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Molecular Spectrum Analyzer</p>', unsafe_allow_html=True)

st.markdown("""
<div class="description-panel">
A remarkable upsurge in the complexity of molecules identified in the interstellar medium (ISM)...
</div>
""", unsafe_allow_html=True)

# === FUNCIONES DE DESCARGA MEJORADAS ===
def download_folder_contents(url, output):
    """Descarga todo el contenido de una carpeta de Google Drive"""
    try:
        # Primero intentamos descargar como zip
        zip_path = os.path.join(output, 'temp_folder.zip')
        os.makedirs(output, exist_ok=True)
        
        # Descargar usando gdown
        gdown.download_folder(url, output=zip_path, quiet=True, use_cookies=False)
        
        # Extraer si es un zip
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output)
            os.remove(zip_path)
        return True
    except Exception as e:
        st.error(f"Error downloading folder contents: {str(e)}")
        return False

def download_resources():
    """Descarga todos los recursos necesarios"""
    MODEL_FOLDER_URL = "https://drive.google.com/drive/folders/1yMH4Ls8f8aCrCS0BTRH57HePUmXWh1AX"
    FILTER_FOLDER_URL = "https://drive.google.com/drive/folders/1s7NaWIyt5mCoiSdBanwPNCekIJ2K65_i"
    
    MODEL_DIR = "downloaded_models"
    FILTER_DIR = "downloaded_filters"
    
    # Limpiar directorios existentes
    shutil.rmtree(MODEL_DIR, ignore_errors=True)
    shutil.rmtree(FILTER_DIR, ignore_errors=True)
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # Descargar modelos (carpeta completa)
    progress_text.text("📥 Downloading models...")
    if not download_folder_contents(MODEL_FOLDER_URL, MODEL_DIR):
        return None, None
    progress_bar.progress(50)
    
    # Descargar filtros (archivos individuales)
    progress_text.text("📥 Downloading filters...")
    if not download_folder_contents(FILTER_FOLDER_URL, FILTER_DIR):
        return None, None
    progress_bar.progress(100)
    
    # Verificar contenido descargado
    model_files = [f for f in os.listdir(MODEL_DIR) if not f.startswith('.')]
    filter_files = [f for f in os.listdir(FILTER_DIR) if f.endswith('.txt')]
    
    if not model_files or not filter_files:
        st.error("No se encontraron archivos válidos en las descargas")
        return None, None
    
    progress_text.text("✅ Download completed successfully!")
    time.sleep(2)
    progress_text.empty()
    progress_bar.empty()
    
    return MODEL_DIR, FILTER_DIR

# === FUNCIONES DE PROCESAMIENTO ===
def read_spectrum(file_path):
    """Lee un archivo de espectro en varios formatos"""
    try:
        if file_path.endswith('.fits'):
            with fits.open(file_path) as hdul:
                data = hdul[1].data
                return data['freq'], data['intensity']
        else:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            data_lines = [line for line in lines if not line.strip().startswith(('!', '//'))]
            data = np.loadtxt(data_lines)
            return data[:, 0], data[:, 1]
    except Exception as e:
        st.error(f"Error reading spectrum file: {str(e)}")
        return None, None

def apply_filter(spectrum_freq, spectrum_intensity, filter_path):
    """Aplica un filtro a un espectro"""
    try:
        filter_data = np.loadtxt(filter_path)
        freq_filter = filter_data[:, 0] / 1e9  # Convertir a GHz
        intensity_filter = filter_data[:, 1]
        
        # Normalizar filtro
        if np.max(intensity_filter) > 0:
            intensity_filter = intensity_filter / np.max(intensity_filter)
        
        # Interpolar espectro a las frecuencias del filtro
        valid_mask = (~np.isnan(spectrum_intensity)) & (~np.isinf(spectrum_intensity))
        interp_func = interp1d(
            spectrum_freq[valid_mask], 
            spectrum_intensity[valid_mask],
            kind='linear', bounds_error=False, fill_value=0
        )
        
        # Aplicar filtro
        filtered = interp_func(freq_filter) * intensity_filter
        filtered = np.clip(filtered, 0, None)
        
        # Crear versión completa con ceros
        full_filtered = np.zeros_like(freq_filter)
        mask = intensity_filter > 0.01  # Umbral para considerar "activado" el filtro
        full_filtered[mask] = filtered[mask]
        
        return {
            'freq': freq_filter,
            'intensity': full_filtered,
            'filter_profile': intensity_filter,
            'mask': mask
        }
    except Exception as e:
        st.error(f"Error applying filter: {str(e)}")
        return None

# === INTERFAZ PRINCIPAL ===
# Descargar recursos al iniciar
MODEL_DIR, FILTER_DIR = download_resources()

# Barra lateral para carga de archivos
st.sidebar.title("Configuration")
input_file = st.sidebar.file_uploader(
    "Input Spectrum File",
    type=['.txt', '.dat', '.fits', '.spec'],
    help="Upload your spectrum file"
)

# Procesamiento principal
if input_file is not None and MODEL_DIR and FILTER_DIR:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(input_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Leer espectro de entrada
        input_freq, input_spec = read_spectrum(tmp_path)
        if input_freq is None:
            raise ValueError("Invalid spectrum file")
        
        # Procesar con todos los filtros
        filter_files = [f for f in os.listdir(FILTER_DIR) if f.endswith('.txt')]
        filtered_results = []
        
        for filter_file in filter_files:
            filter_name = os.path.splitext(filter_file)[0]
            filter_path = os.path.join(FILTER_DIR, filter_file)
            
            result = apply_filter(input_freq, input_spec, filter_path)
            if result is not None:
                # Guardar resultado filtrado
                output_filename = f"filtered_{filter_name}.txt"
                output_path = os.path.join(tempfile.gettempdir(), output_filename)
                np.savetxt(
                    output_path,
                    np.column_stack((result['freq'], result['intensity'])),
                    header=f"Filtered spectrum using {filter_name}",
                    delimiter='\t'
                )
                
                filtered_results.append({
                    'name': filter_name,
                    'original_freq': input_freq,
                    'original_intensity': input_spec,
                    'filtered_data': result,
                    'output_path': output_path
                })
        
        # Mostrar resultados en pestañas
        tab1, tab2 = st.tabs(["Interactive View", "Filter Details"])
        
        with tab1:
            # Gráfico interactivo con todos los filtros
            fig = go.Figure()
            
            # Espectro original
            fig.add_trace(go.Scatter(
                x=input_freq,
                y=input_spec,
                mode='lines',
                name='Original Spectrum',
                line=dict(color='white', width=1.5)))
            
            # Espectros filtrados
            for result in filtered_results:
                fig.add_trace(go.Scatter(
                    x=result['filtered_data']['freq'],
                    y=result['filtered_data']['intensity'],
                    mode='lines',
                    name=f"Filtered: {result['name']}",
                    line=dict(width=1.5)))
            
            fig.update_layout(
                title="Spectrum Filtering Results",
                xaxis_title="Frequency (GHz)",
                yaxis_title="Intensity (K)",
                hovermode="x unified",
                height=600,
                plot_bgcolor='#0D0F14',
                paper_bgcolor='#0D0F14',
                font=dict(color='white'),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Detalles por cada filtro
            for result in filtered_results:
                with st.expander(f"Filter: {result['name']}", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Gráfico del filtro
                        fig_filter = go.Figure()
                        fig_filter.add_trace(go.Scatter(
                            x=result['filtered_data']['freq'],
                            y=result['filtered_data']['filter_profile'],
                            mode='lines',
                            name='Filter Profile',
                            line=dict(color='blue')))
                        fig_filter.update_layout(
                            title="Filter Profile",
                            height=300,
                            plot_bgcolor='#0D0F14',
                            paper_bgcolor='#0D0F14'
                        )
                        st.plotly_chart(fig_filter, use_container_width=True)
                    
                    with col2:
                        # Gráfico comparativo
                        fig_compare = go.Figure()
                        fig_compare.add_trace(go.Scatter(
                            x=result['original_freq'],
                            y=result['original_intensity'],
                            mode='lines',
                            name='Original',
                            line=dict(color='white', width=1)))
                        fig_compare.add_trace(go.Scatter(
                            x=result['filtered_data']['freq'],
                            y=result['filtered_data']['intensity'],
                            mode='lines',
                            name='Filtered',
                            line=dict(color='red', width=1)))
                        fig_compare.update_layout(
                            title="Original vs Filtered",
                            height=300,
                            plot_bgcolor='#0D0F14',
                            paper_bgcolor='#0D0F14'
                        )
                        st.plotly_chart(fig_compare, use_container_width=True)
                    
                    # Botón de descarga
                    with open(result['output_path'], 'rb') as f:
                        st.download_button(
                            label=f"Download {result['name']} filtered spectrum",
                            data=f,
                            file_name=os.path.basename(result['output_path']),
                            mime='text/plain'
                        )
    
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
    finally:
        os.unlink(tmp_path)
elif not MODEL_DIR or not FILTER_DIR:
    st.error("Required resources could not be downloaded. Please try again later.")
else:
    st.info("Please upload a spectrum file to begin analysis")

# Instrucciones en la barra lateral
st.sidebar.markdown("""
**Instructions:**
1. Upload your spectrum file
2. The system will automatically apply all filters
3. View results in the interactive tabs
4. Download filtered spectra as needed
""")
