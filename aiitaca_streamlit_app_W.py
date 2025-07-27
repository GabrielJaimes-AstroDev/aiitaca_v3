import streamlit as st
import os
import numpy as np
import tempfile
import plotly.graph_objects as go
import gdown
from scipy.interpolate import interp1d
from astropy.io import fits
import shutil

# =============================================
# PAGE CONFIGURATION - DEBE SER PRIMERO Y ÚNICO
# =============================================
st.set_page_config(
    layout="wide", 
    page_title="AI-ITACA | Spectrum Analyzer",
    page_icon="🔭"
)

# =============================================
# INITIALIZE SESSION STATE
# =============================================
if not hasattr(st.session_state, 'resources_downloaded'):
    st.session_state.resources_downloaded = False
    st.session_state.MODEL_DIR = None
    st.session_state.FILTER_DIR = None
    st.session_state.downloaded_files = {'models': [], 'filters': []}

# =============================================
# LOAD EXTERNAL RESOURCES
# =============================================
def load_external_file(filename):
    """Carga el contenido de un archivo de texto"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error loading file {filename}: {str(e)}")
        return ""

def load_urls_config(filename):
    """Carga las URLs de configuración con validación robusta"""
    urls = {}
    required_keys = ['MODEL_FOLDER_URL', 'FILTER_FOLDER_URL']
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        urls[key.strip()] = value.strip()
    except Exception as e:
        st.error(f"Error reading config file: {str(e)}")
    
    missing = [key for key in required_keys if key not in urls]
    if missing:
        st.error(f"Missing required URLs: {', '.join(missing)}")
    
    return urls

# Cargar recursos externos
CSS_STYLES = load_external_file('styles_css.txt')
INITIAL_DESCRIPTION = load_external_file('description.txt')
urls = load_urls_config('urls.txt')

MODEL_FOLDER_URL = urls.get('MODEL_FOLDER_URL')
FILTER_FOLDER_URL = urls.get('FILTER_FOLDER_URL')

if not MODEL_FOLDER_URL or not FILTER_FOLDER_URL:
    st.error("Essential configuration missing. Please check urls.txt file.")
    st.stop()

# =============================================
# CSS STYLES
# =============================================
st.markdown(f"<style>{CSS_STYLES}</style>", unsafe_allow_html=True)

# =============================================
# HELPER FUNCTIONS (Optimizadas)
# =============================================
def download_resources():
    """Download all required resources with progress tracking"""
    with st.spinner("🔽 Downloading resources..."):
        MODEL_DIR = "downloaded_models"
        FILTER_DIR = "downloaded_filters"
        
        # Clear existing directories
        shutil.rmtree(MODEL_DIR, ignore_errors=True)
        shutil.rmtree(FILTER_DIR, ignore_errors=True)
        
        # Download models
        if not gdown.download_folder(
            id=MODEL_FOLDER_URL.split('folders/')[-1].split('?')[0],
            output=MODEL_DIR,
            quiet=True,
            use_cookies=False
        ):
            return None, None
        
        # Download filters
        if not gdown.download_folder(
            id=FILTER_FOLDER_URL.split('folders/')[-1].split('?')[0],
            output=FILTER_DIR,
            quiet=True,
            use_cookies=False
        ):
            return None, None
        
        return MODEL_DIR, FILTER_DIR

def robust_read_file(file_path):
    """Improved file reading with better error handling"""
    try:
        # Handle FITS files
        if file_path.endswith('.fits'):
            with fits.open(file_path) as hdul:
                return hdul[1].data['freq'], hdul[1].data['intensity']
        
        # Handle text files
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'ascii']
        for encoding in encodings:
            try:
                lines = content.decode(encoding).splitlines()
                data_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith(('!', '//', '#')):
                        data_lines.append(line.replace(',', '.'))
                
                if not data_lines:
                    continue
                
                data = np.genfromtxt(data_lines)
                if data.ndim == 2 and data.shape[1] >= 2:
                    return data[:, 0], data[:, 1]
            except (UnicodeDecodeError, ValueError):
                continue
        
        raise ValueError("File could not be read with any standard encoding")
    except Exception as e:
        st.error(f"Error reading {os.path.basename(file_path)}: {str(e)}")
        return None, None

def apply_spectral_filter(spectrum_freq, spectrum_intensity, filter_path):
    """Improved spectral filtering with better interpolation"""
    try:
        # Read filter data
        filter_freq, filter_intensity = robust_read_file(filter_path)
        if filter_freq is None:
            return None
        
        # Convert to GHz if needed
        if np.mean(filter_freq) > 1e6:
            filter_freq = filter_freq / 1e9
        
        # Normalize filter
        filter_intensity = filter_intensity / np.max(filter_intensity)
        
        # Create mask for significant regions
        mask = filter_intensity > 0.01
        
        # Validate input spectrum
        valid_points = (~np.isnan(spectrum_intensity)) & (~np.isinf(spectrum_intensity))
        if np.sum(valid_points) < 2:
            raise ValueError("Insufficient valid points in spectrum")
        
        # Improved interpolation
        interp_func = interp1d(
            spectrum_freq[valid_points],
            spectrum_intensity[valid_points],
            kind='linear',
            bounds_error=False,
            fill_value=0.0,
            assume_sorted=False
        )
        
        # Apply filter
        filtered_data = interp_func(filter_freq) * filter_intensity
        filtered_data = np.clip(filtered_data, 0, None)
        
        return {
            'freq': filter_freq,
            'intensity': filtered_data,
            'filter_profile': filter_intensity,
            'mask': mask,
            'filter_name': os.path.splitext(os.path.basename(filter_path))[0],
            'parent_dir': os.path.basename(os.path.dirname(filter_path))
        }
    except Exception as e:
        st.error(f"Filter error {os.path.basename(filter_path)}: {str(e)}")
        return None

def create_spectrum_plot(freq, intensity, name, color, showlegend=True):
    """Helper function to create consistent spectrum plots"""
    return go.Scatter(
        x=freq,
        y=intensity,
        mode='lines',
        name=name,
        line=dict(color=color, width=2),
        showlegend=showlegend
    )

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

st.markdown(f"""
<div class="description-panel">
{INITIAL_DESCRIPTION}
</div>
""", unsafe_allow_html=True)

# =============================================
# MAIN INTERFACE
# =============================================
# Download resources on startup
if not st.session_state.resources_downloaded:
    st.session_state.MODEL_DIR, st.session_state.FILTER_DIR = download_resources()
    if st.session_state.MODEL_DIR and st.session_state.FILTER_DIR:
        st.session_state.resources_downloaded = True
        st.rerun()

# =============================================
# SIDEBAR
# =============================================
st.sidebar.title("Configuration")

# Resource status
with st.sidebar.expander("📁 Resource Status", expanded=True):
    if st.session_state.MODEL_DIR:
        st.success("✅ Models downloaded")
    else:
        st.error("❌ Models missing")
    
    if st.session_state.FILTER_DIR:
        st.success("✅ Filters downloaded")
    else:
        st.error("❌ Filters missing")
    
    if st.button("🔄 Retry Download"):
        st.session_state.resources_downloaded = False
        st.rerun()

# File uploader
input_file = st.sidebar.file_uploader(
    "Upload Spectrum File",
    type=['.txt', '.dat', '.fits', '.spec'],
    help="Supported formats: TXT, DAT, FITS, SPEC"
)

# =============================================
# PROCESSING AND VISUALIZATION
# =============================================
if input_file and st.session_state.resources_downloaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(input_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Read input spectrum
        input_freq, input_spec = robust_read_file(tmp_path)
        if input_freq is None:
            raise ValueError("Failed to read spectrum file")
        
        # Get all filter files
        filter_files = []
        for root, _, files in os.walk(st.session_state.FILTER_DIR):
            for file in files:
                if file.endswith('.txt'):
                    filter_files.append(os.path.join(root, file))
        
        if not filter_files:
            raise ValueError("No filter files found")
        
        # Process filters
        results = []
        with st.spinner("Processing filters..."):
            progress_bar = st.progress(0)
            
            for i, filter_file in enumerate(filter_files):
                progress_bar.progress((i + 1) / len(filter_files))
                result = apply_spectral_filter(input_freq, input_spec, filter_file)
                if result:
                    results.append(result)
            
            progress_bar.empty()
        
        if not results:
            raise ValueError("No filters were successfully applied")
        
        # Create tabs
        tab1, tab2 = st.tabs(["All Filters", "Individual Filters"])
        
        with tab1:
            # Main plot with all filters
            fig = go.Figure()
            fig.add_trace(create_spectrum_plot(
                input_freq, input_spec, "Original", "white"))
            
            for result in results:
                fig.add_trace(create_spectrum_plot(
                    result['freq'], result['intensity'], 
                    result['filter_name'], "#1E88E5"))
            
            fig.update_layout(
                title="All Filtered Spectra",
                xaxis_title="Frequency (GHz)",
                yaxis_title="Intensity (K)",
                height=600,
                plot_bgcolor='#0D0F14',
                paper_bgcolor='#0D0F14',
                font=dict(color='white'),
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Individual filter details
            for result in results:
                with st.expander(f"🔍 {result['filter_name']}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Filter profile
                        fig_profile = go.Figure()
                        fig_profile.add_trace(create_spectrum_plot(
                            result['freq'], result['filter_profile'], 
                            "Filter Profile", "#1E88E5", False))
                        fig_profile.update_layout(
                            title="Filter Profile",
                            height=400,
                            plot_bgcolor='#0D0F14',
                            paper_bgcolor='#0D0F14'
                        )
                        st.plotly_chart(fig_profile, use_container_width=True)
                    
                    with col2:
                        # Comparison
                        fig_comp = go.Figure()
                        fig_comp.add_trace(create_spectrum_plot(
                            input_freq, input_spec, "Original", "white", False))
                        fig_comp.add_trace(create_spectrum_plot(
                            result['freq'], result['intensity'], 
                            "Filtered", "#FF5722", False))
                        fig_comp.update_layout(
                            title="Original vs Filtered",
                            height=400,
                            plot_bgcolor='#0D0F14',
                            paper_bgcolor='#0D0F14'
                        )
                        st.plotly_chart(fig_comp, use_container_width=True)
    
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
    finally:
        os.unlink(tmp_path)

elif not st.session_state.resources_downloaded:
    st.error("Required resources not downloaded. Please check configuration.")
else:
    st.info("Please upload a spectrum file to begin analysis")

# =============================================
# INSTRUCTIONS
# =============================================
st.sidebar.markdown("""
**Instructions:**
1. Upload spectrum file
2. System applies all filters automatically
3. View results in tabs
4. Compare original vs filtered spectra

**Note:** First run may take time to download resources.
""")
