import streamlit as st
import os
import numpy as np
import tempfile
import plotly.graph_objects as go
import time
from scipy.interpolate import interp1d
from astropy.io import fits
import shutil
from glob import glob
import joblib
import pandas as pd
import warnings
from io import StringIO
import matplotlib.pyplot as plt

# =============================================
# CONFIGURACIÓN DE PATHS LOCALES
# =============================================
# Configura estas rutas según tu estructura de directorios
LOCAL_MODEL_DIR = "RF_Models/1.ML_Performance_RDF_CH3OCHO_Noisy_Weight3_Sigma0_001_T1"
LOCAL_FILTER_DIR = "RF_Filters"

# =============================================
# INITIALIZE SESSION STATE
# =============================================
if not hasattr(st.session_state, 'resources_loaded'):
    st.session_state.resources_loaded = False
    st.session_state.MODEL_DIR = None
    st.session_state.FILTER_DIR = None
    st.session_state.local_files = {'models': [], 'filters': []}
    st.session_state.prediction_models_loaded = False
    st.session_state.prediction_results = None

# =============================================
# PAGE CONFIGURATION
# =============================================
st.set_page_config(
    layout="wide", 
    page_title="AI-ITACA | Spectrum Analyzer",
    page_icon="🔭" 
)

# =============================================
# CSS STYLES (from external file)
# =============================================
css_styles = """
<style>
:root {
    --primary-color: #1E88E5;
    --secondary-color: #0D0F14;
    --accent-color: #4CAF50;
    --text-color: #FFFFFF;
    --subtext-color: #BBBBBB;
    --panel-bg: #1E1E1E;
    --error-color: #C62828;
    --warning-color: #FF8F00;
    --success-color: #2E7D32;
}

.main-title {
    font-size: 28px !important;
    font-weight: bold !important;
    color: var(--text-color) !important;
    margin-bottom: 0.2rem !important;
}

.subtitle {
    font-size: 18px !important;
    color: var(--subtext-color) !important;
    margin-bottom: 1.5rem !important;
}

.description-panel {
    background-color: var(--panel-bg);
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
    border-left: 5px solid var(--accent-color);
}

.success-box {
    background-color: var(--success-color);
    color: white;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}

.warning-box {
    background-color: var(--warning-color);
    color: white;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}

.error-box {
    background-color: var(--error-color);
    color: white;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}

.tree-view {
    font-family: monospace;
    margin-left: 15px;
}

.directory {
    color: #4FC3F7;
    margin: 5px 0;
    font-weight: bold;
}

.file {
    color: #E0E0E0;
    margin-left: 20px;
}

.size {
    color: #FF9800;
    font-style: italic;
}

.file-explorer-header {
    font-size: 18px;
    font-weight: bold;
    color: var(--primary-color);
    margin-bottom: 10px;
    border-bottom: 1px solid #444;
    padding-bottom: 5px;
}

.file-explorer-item {
    padding: 5px;
    border-bottom: 1px dotted #444;
}

.buttons-container {
    display: flex;
    gap: 10px;
    margin-bottom: 20px;
}

.physical-params {
    font-size: 16px;
    margin: 8px 0;
    color: var(--text-color);
}

.summary-panel {
    background-color: var(--panel-bg);
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
}

.stButton>button {
    background-color: var(--primary-color);
    color: white;
    border-radius: 5px;
    border: none;
    padding: 8px 16px;
    font-size: 14px;
    transition: all 0.3s;
}

.stButton>button:hover {
    background-color: #1565C0;
    color: white;
}

.stSelectbox>div>div>select {
    background-color: var(--panel-bg);
    color: var(--text-color);
}

.stSlider>div>div>div>div {
    background-color: var(--primary-color);
}

.plotly-container {
    background-color: var(--secondary-color);
    border-radius: 10px;
    padding: 10px;
}

.info-panel {
    background-color: var(--panel-bg);
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 20px;
}
</style>
"""
st.markdown(css_styles, unsafe_allow_html=True)

# =============================================
# HELPER FUNCTIONS
# =============================================
def list_local_files(directory):
    """Recursively list all local files with detailed information"""
    file_list = []
    try:
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if not file.startswith('.'):  # Ignore hidden files
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, directory)
                    try:
                        size = get_file_size(full_path)
                        file_list.append({
                            'path': rel_path,
                            'size': size,
                            'full_path': full_path,
                            'is_file': True,
                            'parent_dir': os.path.basename(root)
                        })
                    except Exception as e:
                        file_list.append({
                            'path': rel_path,
                            'size': 'Error',
                            'full_path': full_path,
                            'is_file': True,
                            'parent_dir': os.path.basename(root)
                        })
    except Exception as e:
        st.error(f"Error listing files in {directory}: {str(e)}")
    return file_list

def get_file_size(path):
    """Get file size in human-readable format"""
    size = os.path.getsize(path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def display_directory_tree(directory, max_depth=3, current_depth=0):
    """Display directory structure as a tree"""
    if not os.path.exists(directory):
        return f"<div class='error-box'>Directory not found: {directory}</div>"
    
    tree_html = "<div class='tree-view'>"
    
    try:
        items = sorted(os.listdir(directory))
        for item in items:
            if item.startswith('.'):  # Ignore hidden files
                continue
                
            full_path = os.path.join(directory, item)
            if os.path.isdir(full_path):
                tree_html += f"<div class='directory'>📁 {item}</div>"
                if current_depth < max_depth:
                    tree_html += display_directory_tree(full_path, max_depth, current_depth+1)
            else:
                size = get_file_size(full_path)
                tree_html += f"<div class='file'>📄 {item} <span class='size'>{size}</span></div>"
    except Exception as e:
        tree_html += f"<div class='error-box'>Error reading directory: {str(e)}</div>"
    
    tree_html += "</div>"
    return tree_html

def robust_read_file(file_path):
    """Read spectrum or filter files with robust format handling"""
    try:
        # FITS files
        if file_path.lower().endswith('.fits'):
            with fits.open(file_path) as hdul:
                if len(hdul) > 1:
                    return hdul[1].data['freq'], hdul[1].data['intensity']
                else:
                    # Handle simple FITS with just data
                    data = hdul[0].data
                    if data.ndim == 2:
                        return data[0], data[1]
                    else:
                        raise ValueError("Unsupported FITS format")
        
        # SPEC files (similar to text but with possible headers)
        if file_path.lower().endswith('.spec'):
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'ascii']:
                try:
                    decoded = content.decode(encoding)
                    lines = decoded.splitlines()
                    break
                except UnicodeDecodeError:
                    continue
            
            data_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith(('!', '//', '#')):
                    cleaned = stripped.replace(',', '.')
                    parts = cleaned.split()
                    if len(parts) >= 2:
                        try:
                            float(parts[0]), float(parts[1])
                            data_lines.append(cleaned)
                        except ValueError:
                            continue
            
            if not data_lines:
                raise ValueError("No valid data found in SPEC file")
            
            data = np.genfromtxt(data_lines)
            return data[:, 0], data[:, 1]
        
        # Text files (TXT, DAT)
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'ascii']:
            try:
                decoded = content.decode(encoding)
                lines = decoded.splitlines()
                
                # Filter comment lines
                data_lines = []
                for line in lines:
                    stripped = line.strip()
                    if stripped and not stripped.startswith(('!', '//', '#')):
                        cleaned = stripped.replace(',', '.')
                        parts = cleaned.split()
                        if len(parts) >= 2:
                            try:
                                float(parts[0]), float(parts[1])
                                data_lines.append(cleaned)
                            except ValueError:
                                continue
                
                if not data_lines:
                    continue
                
                # Read numerical data
                data = np.genfromtxt(data_lines)
                if data.ndim == 2 and data.shape[1] >= 2:
                    return data[:, 0], data[:, 1]
                
            except (UnicodeDecodeError, ValueError):
                continue
        
        raise ValueError("Could not read the file with any standard encoding")
    
    except Exception as e:
        st.error(f"Error reading file {os.path.basename(file_path)}: {str(e)}")
        return None, None

def apply_spectral_filter(spectrum_freq, spectrum_intensity, filter_path):
    """Apply spectral filter with robust handling"""
    try:
        # Read filter data
        filter_freq, filter_intensity = robust_read_file(filter_path)
        if filter_freq is None:
            return None
        
        # Convert to GHz if needed
        if np.mean(filter_freq) > 1e6:
            filter_freq = filter_freq / 1e9
        
        # Normalize filter
        max_intensity = np.max(filter_intensity)
        if max_intensity > 0:
            filter_intensity = filter_intensity / max_intensity
        
        # Create mask for significant regions
        mask = filter_intensity > 0.01
        
        # Validate input spectrum
        valid_points = (~np.isnan(spectrum_intensity)) & (~np.isinf(spectrum_intensity))
        if np.sum(valid_points) < 2:
            raise ValueError("Spectrum doesn't have enough valid points")
        
        # Interpolation
        interp_func = interp1d(
            spectrum_freq[valid_points],
            spectrum_intensity[valid_points],
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # Apply filter
        filtered_data = interp_func(filter_freq) * filter_intensity
        filtered_data = np.clip(filtered_data, 0, None)
        
        # Full version with zeros
        full_filtered = np.zeros_like(filter_freq)
        full_filtered[mask] = filtered_data[mask]
        
        return {
            'freq': filter_freq,
            'intensity': full_filtered,
            'filter_profile': filter_intensity,
            'mask': mask,
            'filter_name': os.path.splitext(os.path.basename(filter_path))[0],
            'parent_dir': os.path.basename(os.path.dirname(filter_path))
        }
    
    except Exception as e:
        st.error(f"Error applying filter {os.path.basename(filter_path)}: {str(e)}")
        return None

def display_file_explorer(files, title, file_type='models'):
    """Display an interactive file explorer"""
    st.markdown(f"<div class='file-explorer-header'>{title}</div>", unsafe_allow_html=True)
    
    # Group files by directory
    dir_structure = {}
    for file in files:
        dir_name = file['parent_dir']
        if dir_name not in dir_structure:
            dir_structure[dir_name] = []
        dir_structure[dir_name].append(file)
    
    # Display as expandable sections
    with st.container():
        for dir_name, dir_files in dir_structure.items():
            with st.expander(f"📁 {dir_name}", expanded=False):
                for file in dir_files:
                    st.markdown(f"""
                    <div class='file-explorer-item'>
                        <b>{os.path.basename(file['path'])}</b>
                        <span style='float: right; color: #FF9800;'>{file['size']}</span>
                    </div>
                    """, unsafe_allow_html=True)

# =============================================
# PREDICTION FUNCTIONS
# =============================================
def load_prediction_models(model_dir):
    """Busca recursivamente los archivos de modelos"""
    required_files = {
        'rf_tex': 'random_forest_tex.pkl',
        'rf_logn': 'random_forest_logn.pkl',
        'x_scaler': 'x_scaler.pkl',
        'tex_scaler': 'tex_scaler.pkl',
        'logn_scaler': 'logn_scaler.pkl'
    }

    try:
        # Buscar archivos recursivamente
        found_files = {}
        for root, _, files in os.walk(model_dir):
            for filename in files:
                for key, req_file in required_files.items():
                    if filename == req_file and key not in found_files:
                        found_files[key] = os.path.join(root, filename)
        
        # Verificar si se encontraron todos
        missing_files = [req_file for key, req_file in required_files.items() if key not in found_files]
        if missing_files:
            st.error(f"Archivos no encontrados: {', '.join(missing_files)}")
            st.error(f"Buscado en: {os.path.abspath(model_dir)}")
            return None, None, None, None, None

        # Cargar modelos con manejo de versiones
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rf_tex = joblib.load(found_files['rf_tex'])
            rf_logn = joblib.load(found_files['rf_logn'])
            x_scaler = joblib.load(found_files['x_scaler'])
            tex_scaler = joblib.load(found_files['tex_scaler'])
            logn_scaler = joblib.load(found_files['logn_scaler'])

        return rf_tex, rf_logn, x_scaler, tex_scaler, logn_scaler

    except Exception as e:
        st.error(f"Error cargando modelos: {str(e)}")
        return None, None, None, None, None

def process_spectrum_for_prediction(file_path, interpolation_length=64610, min_required_points=1000):
    """Process spectrum for prediction"""
    try:
        # Read file with robust reader
        freq, intensity = robust_read_file(file_path)
        if freq is None or intensity is None:
            raise ValueError("Could not read spectrum data")
        
        if len(freq) < min_required_points:
            raise ValueError(f"Spectrum has too few points ({len(freq)})")
        
        # Convert to numpy arrays if not already
        freq = np.array(freq)
        intensity = np.array(intensity)
        
        # Remove NaN and infinite values
        valid = (~np.isnan(freq)) & (~np.isnan(intensity)) & (~np.isinf(intensity))
        freq = freq[valid]
        intensity = intensity[valid]
        
        if len(freq) < min_required_points:
            raise ValueError(f"After cleaning, spectrum has too few points ({len(freq)})")
        
        min_freq = np.min(freq)
        max_freq = np.max(freq)
        freq_range = max_freq - min_freq
        if freq_range == 0:
            raise ValueError("Zero frequency range in test spectrum")
        
        normalized_freq = (freq - min_freq) / freq_range
        
        interp_func = interp1d(normalized_freq, intensity, 
                              kind='linear', bounds_error=False, 
                              fill_value=(intensity[0], intensity[-1]))
        new_freq = np.linspace(0, 1, interpolation_length)
        interpolated_intensity = interp_func(new_freq).astype(np.float32)
        
        # Handle any remaining NaN values
        if np.any(np.isnan(interpolated_intensity)):
            nan_indices = np.where(np.isnan(interpolated_intensity))[0]
            for idx in nan_indices:
                left_idx = max(0, idx-1)
                right_idx = min(interpolation_length-1, idx+1)
                interpolated_intensity[idx] = np.mean(interpolated_intensity[[left_idx, right_idx]])
        
        # Normalize intensity
        min_intensity = np.min(interpolated_intensity)
        max_intensity = np.max(interpolated_intensity)
        if max_intensity != min_intensity:
            scaled_intensity = (interpolated_intensity - min_intensity) / (max_intensity - min_intensity)
        else:
            scaled_intensity = np.zeros_like(interpolated_intensity)
        
        return scaled_intensity
    except Exception as e:
        st.error(f"Error processing test spectrum: {str(e)}")
        return None

def plot_prediction_results(tex_pred, logn_pred):
    """Plot the prediction results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Custom style
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#0D0F14')
    ax1.set_facecolor('#0D0F14')
    ax2.set_facecolor('#0D0F14')
    
    # LogN plot
    ax1.scatter(18.1857, logn_pred, c='red', s=200, edgecolors='black')
    ax1.annotate(f"Pred: {logn_pred:.2f}", 
                (18.1857, logn_pred),
                textcoords="offset points",
                xytext=(15,15), ha='center', fontsize=12, color='red')
    ax1.set_xlabel('LogN de referencia', color='white')
    ax1.set_ylabel('LogN predicho', color='white')
    ax1.set_title('Predicción de LogN', color='white')
    ax1.tick_params(colors='white')
    
    # Tex plot
    ax2.scatter(203.492, tex_pred, c='red', s=200, edgecolors='black')
    ax2.annotate(f"Pred: {tex_pred:.1f}", 
                (203.492, tex_pred),
                textcoords="offset points",
                xytext=(15,15), ha='center', fontsize=12, color='red')
    ax2.set_xlabel('Tex de referencia (K)', color='white')
    ax2.set_ylabel('Tex predicho (K)', color='white')
    ax2.set_title('Predicción de Tex', color='white')
    ax2.tick_params(colors='white')
    
    plt.tight_layout()
    return fig

def run_prediction(filtered_file_path, model_dir):
    """Run the full prediction pipeline with progress bar"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Loading models
    status_text.text("Loading prediction models...")
    rf_tex, rf_logn, x_scaler, tex_scaler, logn_scaler = load_prediction_models(model_dir)
    progress_bar.progress(20)
    
    if None in [rf_tex, rf_logn, x_scaler, tex_scaler, logn_scaler]:
        return None, None
    
    # Step 2: Processing spectrum
    status_text.text("Processing spectrum for prediction...")
    scaled_spectrum = process_spectrum_for_prediction(filtered_file_path)
    progress_bar.progress(50)
    
    if scaled_spectrum is None:
        return None, None
    
    # Step 3: Scaling spectrum
    status_text.text("Preparing data for prediction...")
    scaled_spectrum = x_scaler.transform([scaled_spectrum])
    progress_bar.progress(70)
    
    # Step 4: Making predictions
    status_text.text("Making predictions...")
    tex_pred_scaled = rf_tex.predict(scaled_spectrum)
    tex_pred = tex_scaler.inverse_transform(tex_pred_scaled.reshape(-1, 1))[0,0]
    
    logn_pred_scaled = rf_logn.predict(scaled_spectrum)
    logn_pred = logn_scaler.inverse_transform(logn_pred_scaled.reshape(-1, 1))[0,0]
    
    progress_bar.progress(100)
    status_text.text("Prediction completed!")
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    return tex_pred, logn_pred

# =============================================
# HEADER
# =============================================
st.image("NGC6523_BVO_2.jpg", use_column_width=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.empty()
    
with col2:
    st.markdown('<p class="main-title">AI-ITACA | Artificial Intelligence Integral Tool for AstroChemical Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Molecular Spectrum Analyzer</p>', unsafe_allow_html=True)

# Descripción integrada en el código
description_content = """
<div class="info-panel">
<h3 style="color: #1E88E5; border-bottom: 2px solid #1E88E5; padding-bottom: 10px;">About AI-ITACA</h3>
<p>AI-ITACA is a powerful tool for analyzing molecular spectra using machine learning models. 
This version uses local models and filters instead of downloading them from cloud storage.</p>

<h3 style="color: #1E88E5; border-bottom: 2px solid #1E88E5; padding-bottom: 10px;">How to Use</h3>
<ol>
    <li>Upload your spectrum file (TXT, DAT, FITS, SPEC format)</li>
    <li>The system will automatically apply all available filters</li>
    <li>View results in the interactive tabs</li>
    <li>Make predictions for CH3OCHO parameters</li>
    <li>Download filtered spectra as needed</li>
</ol>

<h3 style="color: #1E88E5; border-bottom: 2px solid #1E88E5; padding-bottom: 10px;">Supported Formats</h3>
<ul>
    <li><strong>TXT/DAT</strong>: Plain text files with two columns (frequency, intensity)</li>
    <li><strong>FITS</strong>: Standard FITS format with spectral data</li>
    <li><strong>SPEC</strong>: Spectral files with optional headers</li>
</ul>
</div>
"""
st.markdown(description_content, unsafe_allow_html=True)

# =============================================
# MAIN INTERFACE
# =============================================
# Load local resources on startup
if not st.session_state.resources_loaded:
    st.session_state.MODEL_DIR = LOCAL_MODEL_DIR
    st.session_state.FILTER_DIR = LOCAL_FILTER_DIR
    
    # Verify paths exist
    if not os.path.exists(st.session_state.MODEL_DIR):
        st.error(f"Model directory not found: {st.session_state.MODEL_DIR}")
    if not os.path.exists(st.session_state.FILTER_DIR):
        st.warning(f"Filter directory not found: {st.session_state.FILTER_DIR}")
    
    try:
        st.session_state.local_files['models'] = list_local_files(st.session_state.MODEL_DIR)
        st.session_state.local_files['filters'] = list_local_files(st.session_state.FILTER_DIR)
        st.session_state.resources_loaded = True
    except Exception as e:
        st.error(f"Error processing local files: {str(e)}")
        st.session_state.resources_loaded = False

# =============================================
# SIDEBAR - CONFIGURATION
# =============================================
st.sidebar.title("Configuration")

# Show local resources
with st.sidebar:
    st.header("📁 Local Resources")
    
    # Models section
    if st.session_state.MODEL_DIR and os.path.exists(st.session_state.MODEL_DIR):
        st.subheader("Models Directory")
        st.code(f"./{st.session_state.MODEL_DIR}", language="bash")
        
        # Display model files in a compact way
        if st.session_state.local_files['models']:
            st.markdown("**Model files:**")
            model_files_text = "\n".join(
                f"- {file['path']} ({file['size']})" 
                for file in st.session_state.local_files['models']
            )
            st.text_area("Model files list", value=model_files_text, height=150, label_visibility="collapsed")
        else:
            st.warning("No model files found")
    
    # Filters section
    if st.session_state.FILTER_DIR and os.path.exists(st.session_state.FILTER_DIR):
        st.subheader("Filters Directory")
        st.code(f"./{st.session_state.FILTER_DIR}", language="bash")
        
        # Display filter files in a compact way
        if st.session_state.local_files['filters']:
            st.markdown("**Filter files:**")
            filter_files_text = "\n".join(
                f"- {file['path']} ({file['size']})" 
                for file in st.session_state.local_files['filters']
            )
            st.text_area("Filter files list", value=filter_files_text, height=150, label_visibility="collapsed")
        else:
            st.warning("No filter files found")

    # Button to reload resources
    if st.button("🔄 Reload Resources"):
        st.session_state.local_files['models'] = list_local_files(st.session_state.MODEL_DIR)
        st.session_state.local_files['filters'] = list_local_files(st.session_state.FILTER_DIR)
        st.experimental_rerun()

# File selector
input_file = st.sidebar.file_uploader(
    "Input Spectrum File",
    type=['.txt', '.dat', '.fits', '.spec'],
    help="Upload your spectrum file (TXT, DAT, FITS, SPEC)"
)

# Units Configuration
st.sidebar.markdown("---")
st.sidebar.subheader("Units Configuration")

freq_unit = st.sidebar.selectbox(
    "Frequency Units",
    ["GHz", "MHz", "kHz", "Hz"],
    index=0,
    help="Select the frequency units for the input spectrum"
)

intensity_unit = st.sidebar.selectbox(
    "Intensity Units",
    ["K", "Jy"],
    index=0,
    help="Select the intensity units for the input spectrum"
)

# =============================================
# MAIN PROCESSING
# =============================================
if input_file is not None and st.session_state.MODEL_DIR and st.session_state.FILTER_DIR:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(input_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Read input spectrum
        input_freq, input_spec = robust_read_file(tmp_path)
        if input_freq is None:
            raise ValueError("Could not read the spectrum file")
        
        # Get all filter files
        filter_files = []
        for root, _, files in os.walk(st.session_state.FILTER_DIR):
            for file in files:
                if file.endswith('.txt'):
                    filter_files.append(os.path.join(root, file))
        
        if not filter_files:
            raise ValueError("No filter files found in the filters directory")
        
        # Process with all filters
        filtered_results = []
        failed_filters = []
        
        with st.spinner("🔍 Applying filters..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, filter_file in enumerate(filter_files):
                filter_name = os.path.splitext(os.path.basename(filter_file))[0]
                status_text.text(f"Processing filter {i+1}/{len(filter_files)}: {filter_name}")
                progress_bar.progress((i + 1) / len(filter_files))
                
                result = apply_spectral_filter(input_freq, input_spec, filter_file)
                if result is not None:
                    # Save filtered result temporarily
                    output_filename = f"filtered_{result['filter_name']}.txt"
                    output_path = os.path.join(tempfile.gettempdir(), output_filename)
                    
                    header = f"!xValues(GHz)\tyValues(K)\n!Filter applied: {result['filter_name']}"
                    np.savetxt(
                        output_path,
                        np.column_stack((result['freq'], result['intensity'])),
                        header=header,
                        delimiter='\t',
                        fmt=['%.10f', '%.6e'],
                        comments=''
                    )
                    
                    filtered_results.append({
                        'name': result['filter_name'],
                        'original_freq': input_freq,
                        'original_intensity': input_spec,
                        'filtered_data': result,
                        'output_path': output_path
                    })
                else:
                    failed_filters.append(os.path.basename(filter_file))
            
            progress_bar.empty()
            status_text.empty()
        
        # Show results
        if not filtered_results:
            raise ValueError(f"No filters were successfully applied. {len(failed_filters)} filters failed.")
        
        st.markdown(f'<div class="success-box">✅ Successfully applied {len(filtered_results)} filters</div>', unsafe_allow_html=True)
        
        if failed_filters:
            st.markdown(f'<div class="warning-box">⚠ Failed to apply {len(failed_filters)} filters: {", ".join(failed_filters)}</div>', unsafe_allow_html=True)
        
        # Show in tabs
        tab1, tab2, tab3 = st.tabs(["Interactive Spectrum", "Filter Details", "CH3OCHO Prediction"])
        
        with tab1:
            # Main interactive graph
            fig_main = go.Figure()
            
            # Original spectrum
            fig_main.add_trace(go.Scatter(
                x=input_freq,
                y=input_spec,
                mode='lines',
                name='Original Spectrum',
                line=dict(color='white', width=2))
            )
            
            # Add all filtered spectra
            for result in filtered_results:
                fig_main.add_trace(go.Scatter(
                    x=result['filtered_data']['freq'],
                    y=result['filtered_data']['intensity'],
                    mode='lines',
                    name=f"Filtered: {result['name']}",
                    line=dict(width=1.5))
                )
            
            fig_main.update_layout(
                title="Spectrum Filtering Results",
                xaxis_title=f"Frequency ({freq_unit})",
                yaxis_title=f"Intensity ({intensity_unit})",
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
                ),
                xaxis=dict(gridcolor='#3A3A3A'),
                yaxis=dict(gridcolor='#3A3A3A')
            )
            st.plotly_chart(fig_main, use_column_width=True)
        
        with tab2:
            # Details for each filter
            for result in filtered_results:
                with st.expander(f"Filter: {result['name']} (from {result['filtered_data']['parent_dir']})", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Filter profile
                        fig_filter = go.Figure()
                        fig_filter.add_trace(go.Scatter(
                            x=result['filtered_data']['freq'],
                            y=result['filtered_data']['filter_profile'],
                            mode='lines',
                            name='Filter Profile',
                            line=dict(color='#1E88E5'))
                        )
                        fig_filter.update_layout(
                            title="Filter Profile",
                            height=300,
                            plot_bgcolor='#0D0F14',
                            paper_bgcolor='#0D0F14',
                            showlegend=False,
                            font=dict(color='white'),
                            xaxis=dict(gridcolor='#3A3A3A'),
                            yaxis=dict(gridcolor='#3A3A3A')
                        )
                        st.plotly_chart(fig_filter, use_column_width=True)
                    
                    with col2:
                        # Original vs filtered comparison
                        fig_compare = go.Figure()
                        fig_compare.add_trace(go.Scatter(
                            x=result['original_freq'],
                            y=result['original_intensity'],
                            mode='lines',
                            name='Original',
                            line=dict(color='white', width=1))
                        )
                        fig_compare.add_trace(go.Scatter(
                            x=result['filtered_data']['freq'],
                            y=result['filtered_data']['intensity'],
                            mode='lines',
                            name='Filtered',
                            line=dict(color='#FF5722', width=1))
                        )
                        fig_compare.update_layout(
                            title="Original vs Filtered",
                            height=300,
                            plot_bgcolor='#0D0F14',
                            paper_bgcolor='#0D0F14',
                            showlegend=False,
                            font=dict(color='white'),
                            xaxis=dict(gridcolor='#3A3A3A'),
                            yaxis=dict(gridcolor='#3A3A3A')
                        )
                        st.plotly_chart(fig_compare, use_column_width=True)
                    
                    # Download button
                    with open(result['output_path'], 'rb') as f:
                        st.download_button(
                            label=f"Download {result['name']} filtered spectrum",
                            data=f,
                            file_name=os.path.basename(result['output_path']),
                            mime='text/plain',
                            key=f"download_{result['name']}",
                            use_container_width=True
                        )
        
        with tab3:
            # CH3OCHO Prediction Tab
            st.markdown("<h2 style='text-align: center;'>CH3OCHO Spectral Prediction</h2>", unsafe_allow_html=True)
            
            # Find the CH3OCHO filtered file
            ch3ocho_result = None
            for result in filtered_results:
                if "CH3OCHO" in result['name'].upper():
                    ch3ocho_result = result
                    break
            
            if ch3ocho_result:
                st.success(f"Found CH3OCHO filtered spectrum: {ch3ocho_result['name']}")
                
                if st.button("Run CH3OCHO Prediction"):
                    with st.spinner("Running prediction for CH3OCHO..."):
                        tex_pred, logn_pred = run_prediction(ch3ocho_result['output_path'], st.session_state.MODEL_DIR)
                    
                    if tex_pred is not None and logn_pred is not None:
                        st.session_state.prediction_results = {
                            'tex': tex_pred,
                            'logn': logn_pred
                        }
                        
                        st.markdown("### Prediction Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(label="Predicted Tex", value=f"{tex_pred:.2f} K")
                        
                        with col2:
                            st.metric(label="Predicted LogN", value=f"{logn_pred:.2f}")
                        
                        # Plot results
                        st.markdown("### Prediction Visualization")
                        fig = plot_prediction_results(tex_pred, logn_pred)
                        st.pyplot(fig)
                        
                        # Download results
                        st.download_button(
                            label="Download Prediction Results",
                            data=f"Tex: {tex_pred:.2f} K\nLogN: {logn_pred:.2f}",
                            file_name="ch3ocho_prediction_results.txt",
                            mime='text/plain'
                        )
                    else:
                        st.error("Prediction failed. Please check the logs for errors.")
            else:
                st.warning("No CH3OCHO filtered spectrum found. Please ensure the CH3OCHO filter was applied successfully.")
    
    except Exception as e:
        st.markdown(f'<div class="error-box">❌ Processing error: {str(e)}</div>', unsafe_allow_html=True)
    finally:
        os.unlink(tmp_path)

elif not st.session_state.MODEL_DIR or not st.session_state.FILTER_DIR:
    st.markdown("""
    <div class="error-box">
    ❌ Required local resources not found.<br><br>
    Please ensure you have:
    <ol>
        <li>Created a 'RF_Models' directory with the model files</li>
        <li>Created a 'RF_Filters' directory with filter files</li>
        <li>Placed them in the correct location</li>
    </ol>
    Click the 'Reload Resources' button in the sidebar after setting up the directories.
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("ℹ️ Please upload a spectrum file to begin analysis")

# =============================================
# INSTRUCTIONS
# =============================================
st.sidebar.markdown("""
**Instructions:**
1. Upload your spectrum file
2. The system will automatically apply all filters
3. View results in the interactive tabs
4. Download filtered spectra as needed

**Supported formats:**
- Text files (.txt, .dat)
- FITS files (.fits)
- Spectrum files (.spec)

**Interactive Plot Controls:**
- 🔍 Zoom: Click and drag to select area
- 🖱️ Hover: View exact values
- 🔄 Reset: Double-click
- 🏎️ Pan: Shift+click+drag
""")
