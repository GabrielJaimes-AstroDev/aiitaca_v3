import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import plotly.graph_objects as go
import tensorflow as tf
import gdown
import shutil
import time
from scipy.interpolate import interp1d
from glob import glob
from astropy.io import fits
import zipfile

st.set_page_config(
    layout="wide", 
    page_title="AI-ITACA | Spectrum Analyzer",
    page_icon="🔭" 
)

# === CUSTOM CSS STYLES ===
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
        transition: all 0.3s !important;
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

# === HEADER WITH IMAGE AND DESCRIPTION ===
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

# === DOWNLOAD FUNCTIONS ===
def download_and_extract_models():
    MODEL_FOLDER_URL = "https://drive.google.com/drive/folders/1yMH4Ls8f8aCrCS0BTRH57HePUmXWh1AX"
    FILTER_FOLDER_URL = "https://drive.google.com/drive/folders/1s7NaWIyt5mCoiSdBanwPNCekIJ2K65_i"
    MODEL_DIR = "downloaded_models"
    FILTER_DIR = "downloaded_filters"
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FILTER_DIR, exist_ok=True)
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # Download models
    progress_text.text("📥 Downloading models...")
    gdown.download_folder(MODEL_FOLDER_URL, output=MODEL_DIR, quiet=True)
    progress_bar.progress(30)
    
    # Download filters
    progress_text.text("📥 Downloading filters...")
    gdown.download_folder(FILTER_FOLDER_URL, output=FILTER_DIR, quiet=True)
    progress_bar.progress(70)
    
    # Verify downloads
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
    filter_files = [f for f in os.listdir(FILTER_DIR) if f.endswith('.txt')]
    
    if model_files and filter_files:
        progress_text.text("✅ Download completed successfully!")
        progress_bar.progress(100)
        time.sleep(2)
        progress_text.empty()
        progress_bar.empty()
        return MODEL_DIR, FILTER_DIR
    else:
        st.error("Failed to download required files")
        return None, None

# === SPECTRUM PROCESSING FUNCTIONS ===
def read_spectrum(file_path):
    if file_path.endswith('.fits'):
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            freq = data['freq']
            intensity = data['intensity']
    else:  # .txt, .dat, etc.
        with open(file_path, 'r') as f:
            lines = f.readlines()
        data_lines = [line for line in lines if not (line.strip().startswith('!') or line.strip().startswith('//')]
        data = np.loadtxt(data_lines)
        freq = data[:, 0]
        intensity = data[:, 1]
    return freq, intensity

def apply_filter(spectrum_freq, spectrum_intensity, filter_path):
    filter_data = np.loadtxt(filter_path, comments='/')
    freq_filter_hz = filter_data[:, 0]
    intensity_filter = filter_data[:, 1]
    freq_filter = freq_filter_hz / 1e9  # Convert to GHz
    
    # Normalize filter
    if np.max(intensity_filter) > 0:
        intensity_filter = intensity_filter / np.max(intensity_filter)
    
    # Interpolate spectrum to filter frequencies
    interp_func = interp1d(spectrum_freq, spectrum_intensity, 
                          kind='cubic', bounds_error=False, fill_value=np.nan)
    spectrum_interpolated = interp_func(freq_filter)
    
    # Apply filter
    mask = intensity_filter != 0
    filtered_freqs = freq_filter[mask]
    filtered_intensities = spectrum_interpolated[mask]
    filtered_intensities = np.clip(filtered_intensities, 0, None)
    
    return filtered_freqs, filtered_intensities, freq_filter, intensity_filter

# === MAIN APP ===
MODEL_DIR, FILTER_DIR = download_and_extract_models()

# Sidebar configuration
st.sidebar.title("Configuration")
input_file = st.sidebar.file_uploader(
    "Input Spectrum File ( . | .txt | .dat | .fits | .spec )",
    type=None,
    help="Drag and drop file here ( . | .txt | .dat | .fits | .spec ). Limit 200MB per file"
)

if input_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
        tmp_file.write(input_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Read input spectrum
        input_freq, input_spec = read_spectrum(tmp_path)
        
        # Process with all filters
        filter_files = [f for f in os.listdir(FILTER_DIR) if f.endswith('.txt')]
        filtered_results = []
        
        for filter_file in filter_files:
            filter_path = os.path.join(FILTER_DIR, filter_file)
            filtered_freqs, filtered_intensities, freq_filter, intensity_filter = apply_filter(
                input_freq, input_spec, filter_path)
            
            # Save filtered spectrum
            filter_name = os.path.splitext(filter_file)[0]
            output_filename = f"filtered_{filter_name}.txt"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            
            header = f"!xValues(GHz)\tyValues(K)\n!Filter: {filter_name}"
            np.savetxt(output_path,
                      np.column_stack((filtered_freqs, filtered_intensities)),
                      header=header,
                      delimiter='\t',
                      fmt=['%.10f', '%.6e'],
                      comments='')
            
            filtered_results.append({
                'name': filter_name,
                'original_freq': input_freq,
                'original_intensity': input_spec,
                'filter_freq': freq_filter,
                'filter_intensity': intensity_filter,
                'filtered_freq': filtered_freqs,
                'filtered_intensity': filtered_intensities,
                'output_path': output_path
            })
        
        # Display results in tabs
        tab1, tab2 = st.tabs(["Interactive View", "Filter Details"])
        
        with tab1:
            fig = go.Figure()
            
            # Add original spectrum
            fig.add_trace(go.Scatter(
                x=input_freq,
                y=input_spec,
                mode='lines',
                name='Original Spectrum',
                line=dict(color='white', width=2))
            
            # Add filtered spectra
            for result in filtered_results:
                fig.add_trace(go.Scatter(
                    x=result['filtered_freq'],
                    y=result['filtered_intensity'],
                    mode='lines',
                    name=f"Filtered: {result['name']}",
                    line=dict(width=2))
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
        
        with tab2:
            for result in filtered_results:
                with st.expander(f"Filter: {result['name']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Original vs Filtered**")
                        fig1 = go.Figure()
                        fig1.add_trace(go.Scatter(
                            x=result['original_freq'],
                            y=result['original_intensity'],
                            mode='lines',
                            name='Original',
                            line=dict(color='white')))
                        fig1.add_trace(go.Scatter(
                            x=result['filtered_freq'],
                            y=result['filtered_intensity'],
                            mode='lines',
                            name='Filtered',
                            line=dict(color='red')))
                        fig1.update_layout(height=300)
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Filter Profile**")
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(
                            x=result['filter_freq'],
                            y=result['filter_intensity'],
                            mode='lines',
                            line=dict(color='blue')))
                        fig2.update_layout(height=300)
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    st.download_button(
                        label=f"Download Filtered Spectrum ({result['name']})",
                        data=open(result['output_path'], 'rb').read(),
                        file_name=os.path.basename(result['output_path']),
                        mime='text/plain'
                    )
    
    except Exception as e:
        st.error(f"Error processing spectrum: {str(e)}")
    finally:
        os.unlink(tmp_path)
else:
    st.info("Please upload an input spectrum file to begin analysis.")

# Instructions in sidebar
st.sidebar.markdown("""
**Instructions:**
1. Upload your input spectrum file
2. The system will automatically apply all filters
3. View results in the interactive tabs
""")
