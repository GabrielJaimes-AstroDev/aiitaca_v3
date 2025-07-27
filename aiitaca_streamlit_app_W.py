import streamlit as st
import os
import numpy as np
import tempfile
import plotly.graph_objects as go
import gdown
import time
from scipy.interpolate import interp1d
from astropy.io import fits
import shutil
from glob import glob

# =============================================
# INITIALIZE SESSION STATE
# =============================================
if not hasattr(st.session_state, 'resources_downloaded'):
    st.session_state.resources_downloaded = False
    st.session_state.MODEL_DIR = None
    st.session_state.FILTER_DIR = None
    st.session_state.downloaded_files = {'models': [], 'filters': []}

# =============================================
# PAGE CONFIGURATION
# =============================================
st.set_page_config(
    layout="wide", 
    page_title="AI-ITACA | Spectrum Analyzer",
    page_icon="🔭"
)

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
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' not in line:
                    st.warning(f"Formato inválido en {filename}, línea {line_number}: '{line}' - debe ser 'clave=valor'")
                    continue
                
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if not value:
                    st.warning(f"Valor vacío en {filename}, línea {line_number} para clave '{key}'")
                    continue
                
                urls[key] = value
    
    except FileNotFoundError:
        st.error(f"Archivo de configuración no encontrado: {filename}")
    except Exception as e:
        st.error(f"Error inesperado leyendo {filename}: {str(e)}")
    
    missing_keys = [key for key in required_keys if key not in urls]
    if missing_keys:
        st.error(f"Faltan URLs requeridas en {filename}: {', '.join(missing_keys)}")
    
    return urls

# Cargar recursos externos
CSS_STYLES = load_external_file('styles_css.txt')
INITIAL_DESCRIPTION = load_external_file('description.txt')

# Cargar configuración de URLs
urls = load_urls_config('urls.txt')
MODEL_FOLDER_URL = urls.get('MODEL_FOLDER_URL')
FILTER_FOLDER_URL = urls.get('FILTER_FOLDER_URL')

# Validación crítica
if not MODEL_FOLDER_URL or not FILTER_FOLDER_URL:
    st.error("""
    ❌ Configuración esencial faltante. La aplicación no puede continuar.
    Por favor verifique que el archivo urls.txt contiene:
    MODEL_FOLDER_URL=su_url_aqui
    FILTER_FOLDER_URL=su_url_aqui
    """)
    st.stop()

# =============================================
# CSS STYLES
# =============================================
st.markdown(f"<style>{CSS_STYLES}</style>", unsafe_allow_html=True)

# =============================================
# HELPER FUNCTIONS
# =============================================
def list_downloaded_files(directory):
    """Recursively list all downloaded files with detailed information"""
    file_list = []
    try:
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if not file.startswith('.'):
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

def download_google_drive_folder(folder_url, output_dir):
    """Recursively download all content from a Google Drive folder"""
    try:
        folder_id = folder_url.split('folders/')[-1].split('?')[0]
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
        
        gdown.download_folder(
            id=folder_id,
            output=output_dir,
            quiet=True,
            use_cookies=False,
            remaining_ok=True
        )
        return True
    except Exception as e:
        st.error(f"Error downloading folder: {str(e)}")
        return False

def download_resources():
    """Download all required resources"""
    with st.spinner("🔽 Downloading models (this may take several minutes)..."):
        if not download_google_drive_folder(MODEL_FOLDER_URL, "downloaded_models"):
            return None, None
    
    with st.spinner("🔽 Downloading filters..."):
        if not download_google_drive_folder(FILTER_FOLDER_URL, "downloaded_filters"):
            return None, None
    
    return "downloaded_models", "downloaded_filters"

def robust_read_file(file_path):
    """Read spectrum or filter files with robust format handling"""
    try:
        # FITS files
        if file_path.endswith('.fits'):
            with fits.open(file_path) as hdul:
                return hdul[1].data['freq'], hdul[1].data['intensity']
        
        # Text files
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
                        data_lines.append(cleaned)
                
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
        if filter_freq is None or filter_intensity is None:
            return None
        
        # Ensure frequencies are in GHz
        if np.mean(filter_freq) > 1e6:  # If in Hz, convert to GHz
            filter_freq = filter_freq / 1e9
        
        # Normalize filter profile
        filter_intensity = filter_intensity / np.max(filter_intensity)
        
        # Create mask for significant regions
        significant_mask = filter_intensity > 0.01
        
        # Validate input spectrum
        valid_points = (~np.isnan(spectrum_intensity)) & (~np.isinf(spectrum_intensity))
        if np.sum(valid_points) < 2:
            raise ValueError("Spectrum doesn't have enough valid points")
        
        # Interpolate spectrum to filter frequencies
        interp_func = interp1d(
            spectrum_freq[valid_points],
            spectrum_intensity[valid_points],
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # Apply filter by multiplying spectrum with filter profile
        filtered_intensity = interp_func(filter_freq) * filter_intensity
        
        # Create full output with zeros outside significant regions
        full_filtered = np.zeros_like(filter_freq)
        full_filtered[significant_mask] = filtered_intensity[significant_mask]
        
        return {
            'freq': filter_freq,
            'intensity': full_filtered,
            'filter_profile': filter_intensity,
            'mask': significant_mask,
            'filter_name': os.path.splitext(os.path.basename(filter_path))[0],
            'parent_dir': os.path.basename(os.path.dirname(filter_path))
        }
    
    except Exception as e:
        st.error(f"Error applying filter {os.path.basename(filter_path)}: {str(e)}")
        return None

def display_file_explorer(files, title, file_type='models'):
    """Display an interactive file explorer"""
    st.markdown(f"<div class='file-explorer-header'>{title}</div>", unsafe_allow_html=True)
    
    dir_structure = {}
    for file in files:
        dir_name = file['parent_dir']
        if dir_name not in dir_structure:
            dir_structure[dir_name] = []
        dir_structure[dir_name].append(file)
    
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
if not st.session_state.resources_downloaded:
    st.session_state.MODEL_DIR, st.session_state.FILTER_DIR = download_resources()
    if st.session_state.MODEL_DIR and st.session_state.FILTER_DIR:
        try:
            st.session_state.downloaded_files['models'] = list_downloaded_files(st.session_state.MODEL_DIR)
            st.session_state.downloaded_files['filters'] = list_downloaded_files(st.session_state.FILTER_DIR)
            st.session_state.resources_downloaded = True
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Error processing downloaded files: {str(e)}")
            st.session_state.resources_downloaded = False

# =============================================
# SIDEBAR - CONFIGURATION
# =============================================
st.sidebar.title("Configuration")

with st.sidebar:
    with st.expander("📁 Downloaded Resources", expanded=False):
        st.markdown('<div class="resource-panel">', unsafe_allow_html=True)
        
        if st.session_state.MODEL_DIR and os.path.exists(st.session_state.MODEL_DIR):
            st.subheader("Models Directory")
            st.code(f"./{st.session_state.MODEL_DIR}", language="bash")
            
            if st.session_state.downloaded_files['models']:
                st.markdown("**Model files:**")
                model_files_text = "\n".join(
                    f"- {file['path']} ({file['size']})" 
                    for file in st.session_state.downloaded_files['models']
                )
                st.text_area("Model files list", value=model_files_text, height=150, label_visibility="collapsed")
            else:
                st.warning("No model files found")
        
        if st.session_state.FILTER_DIR and os.path.exists(st.session_state.FILTER_DIR):
            st.subheader("Filters Directory")
            st.code(f"./{st.session_state.FILTER_DIR}", language="bash")
            
            if st.session_state.downloaded_files['filters']:
                st.markdown("**Filter files:**")
                filter_files_text = "\n".join(
                    f"- {file['path']} ({file['size']})" 
                    for file in st.session_state.downloaded_files['filters']
                )
                st.text_area("Filter files list", value=filter_files_text, height=150, label_visibility="collapsed")
            else:
                st.warning("No filter files found")

        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔄 Retry Download Resources"):
        st.session_state.MODEL_DIR, st.session_state.FILTER_DIR = download_resources()
        if st.session_state.MODEL_DIR and st.session_state.FILTER_DIR:
            st.session_state.downloaded_files['models'] = list_downloaded_files(st.session_state.MODEL_DIR)
            st.session_state.downloaded_files['filters'] = list_downloaded_files(st.session_state.FILTER_DIR)
            st.session_state.resources_downloaded = True
            st.rerun()

input_file = st.sidebar.file_uploader(
    "Input Spectrum File",
    type=['.txt', '.dat', '.fits', '.spec'],
    help="Upload your spectrum file (TXT, DAT, FITS, SPEC)"
)

# =============================================
# IMPROVED FILTER PROCESSING AND VISUALIZATION
# =============================================
if input_file is not None and st.session_state.MODEL_DIR and st.session_state.FILTER_DIR:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(input_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Read input spectrum with robust handling
        input_freq, input_spec = robust_read_file(tmp_path)
        if input_freq is None or input_spec is None:
            raise ValueError("Could not read the spectrum file")
        
        # Convert input frequencies to GHz if needed
        if np.mean(input_freq) > 1e6:  # If in Hz, convert to GHz
            input_freq = input_freq / 1e9
        
        # Get all filter files recursively
        filter_files = []
        for root, _, files in os.walk(st.session_state.FILTER_DIR):
            for file in files:
                if file.endswith('.txt'):
                    filter_files.append(os.path.join(root, file))
        
        if not filter_files:
            raise ValueError("No filter files found in the filters directory")
        
        # Create directory for filtered spectra
        filtered_dir = os.path.join(tempfile.gettempdir(), "filtered_spectra")
        os.makedirs(filtered_dir, exist_ok=True)
        
        # Process with all filters
        filtered_results = []
        failed_filters = []
        
        with st.spinner("🔍 Applying filters and analyzing results..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, filter_file in enumerate(filter_files):
                filter_name = os.path.splitext(os.path.basename(filter_file))[0]
                status_text.text(f"Processing filter {i+1}/{len(filter_files)}: {filter_name}")
                progress_bar.progress((i + 1) / len(filter_files))
                
                result = apply_spectral_filter(input_freq, input_spec, filter_file)
                if result is not None:
                    # Save filtered result
                    output_filename = f"filtered_{result['filter_name']}.txt"
                    output_path = os.path.join(filtered_dir, output_filename)
                    
                    header = f"Frequency(GHz)\tIntensity(K)\n# Filter applied: {result['filter_name']}"
                    np.savetxt(
                        output_path,
                        np.column_stack((result['freq'], result['intensity'])),
                        header=header,
                        delimiter='\t',
                        fmt=['%.10f', '%.6e'],
                        comments=''
                    )
                    
                    # Calculate additional metrics
                    original_area = np.trapz(input_spec, input_freq)
                    filtered_area = np.trapz(result['intensity'], result['freq'])
                    transmission = filtered_area / original_area if original_area > 0 else 0
                    
                    filtered_results.append({
                        'name': result['filter_name'],
                        'original_freq': input_freq,
                        'original_intensity': input_spec,
                        'filtered_data': result,
                        'output_path': output_path,
                        'transmission': transmission,
                        'parent_dir': result['parent_dir']
                    })
                else:
                    failed_filters.append(os.path.basename(filter_file))
            
            progress_bar.empty()
            status_text.empty()
        
        # Show processing results
        if not filtered_results:
            raise ValueError(f"No filters were successfully applied. {len(failed_filters)} filters failed.")
        
        st.markdown(f'<div class="success-box">✅ Successfully applied {len(filtered_results)} filters</div>', unsafe_allow_html=True)
        
        if failed_filters:
            st.markdown(f'<div class="warning-box">⚠ Failed to apply {len(failed_filters)} filters: {", ".join(failed_filters)}</div>', unsafe_allow_html=True)
        
        # Create tabs for visualization
        tab1, tab2 = st.tabs(["Interactive Spectrum Explorer", "Filter Comparison"])
        
        with tab1:
            # Main interactive graph with all filters
            fig_main = go.Figure()
            
            # Original spectrum
            fig_main.add_trace(go.Scatter(
                x=input_freq,
                y=input_spec,
                mode='lines',
                name='Original Spectrum',
                line=dict(color='white', width=2),
                hoverinfo='x+y'
            ))
            
            # Add all filtered spectra with improved visualization
            colors = ['#1E88E5', '#FF5722', '#4CAF50', '#FFC107', '#9C27B0']
            for i, result in enumerate(filtered_results):
                color = colors[i % len(colors)]
                fig_main.add_trace(go.Scatter(
                    x=result['filtered_data']['freq'],
                    y=result['filtered_data']['intensity'],
                    mode='lines',
                    name=f"{result['name']} (Transmission: {result['transmission']:.2%})",
                    line=dict(color=color, width=1.5),
                    hoverinfo='x+y+name',
                    visible='legendonly' if len(filtered_results) > 5 else True
                ))
            
            fig_main.update_layout(
                title="Spectrum Filtering Results - All Filters",
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
            st.plotly_chart(fig_main, use_container_width=True)
            
            # Add spectrum statistics
            with st.expander("📊 Spectrum Statistics", expanded=False):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Frequency Range", f"{min(input_freq):.2f} - {max(input_freq):.2f} GHz")
                with col2:
                    st.metric("Max Intensity", f"{max(input_spec):.2e} K")
                with col3:
                    st.metric("Total Area", f"{np.trapz(input_spec, input_freq):.2e} K·GHz")
        
        # En la sección donde se crean los botones de descarga (dentro del bucle for que procesa los filtros):
with tab2:
    # Sort filters by transmission efficiency
    filtered_results_sorted = sorted(filtered_results, key=lambda x: x['transmission'], reverse=True)
    
    # Show filter comparison table
    st.markdown("### Filter Performance Comparison")
    comparison_data = []
    for i, result in enumerate(filtered_results_sorted):  # Añadimos enumerate para obtener índice único
        comparison_data.append({
            "Filter Name": result['name'],
            "Category": result['parent_dir'],
            "Transmission": f"{result['transmission']:.2%}",
            "Max Intensity": f"{max(result['filtered_data']['intensity']):.2e} K"
        })
    
    st.table(comparison_data)
    
    # Show details for each filter
    st.markdown("### Filter Details")
    for i, result in enumerate(filtered_results_sorted):  # Añadimos enumerate aquí también
        with st.expander(f"🔍 {result['name']} (Transmission: {result['transmission']:.2%})", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # Filter profile plot
                fig_filter = go.Figure()
                fig_filter.add_trace(go.Scatter(
                    x=result['filtered_data']['freq'],
                    y=result['filtered_data']['filter_profile'],
                    mode='lines',
                    name='Filter Profile',
                    line=dict(color='#1E88E5', width=2))
                )
                fig_filter.update_layout(
                    title="Filter Profile",
                    height=400,
                    plot_bgcolor='#0D0F14',
                    paper_bgcolor='#0D0F14',
                    showlegend=False,
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(color='white')
                )
                st.plotly_chart(fig_filter, use_container_width=True)
            
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
                    line=dict(color='#FF5722', width=2))
                )
                fig_compare.update_layout(
                    title="Original vs Filtered",
                    height=400,
                    plot_bgcolor='#0D0F14',
                    paper_bgcolor='#0D0F14',
                    showlegend=False,
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(color='white')
                )
                st.plotly_chart(fig_compare, use_container_width=True)
            
            # Download button with unique key using the index
            with open(result['output_path'], 'rb') as f:
                st.download_button(
                    label=f"📥 Download {result['name']} filtered spectrum",
                    data=f,
                    file_name=os.path.basename(result['output_path']),
                    mime='text/plain',
                    key=f"download_{result['name']}_{i}",  # Añadimos el índice para hacer la clave única
                    use_container_width=True
                )
    
    except Exception as e:
        st.markdown(f'<div class="error-box">❌ Processing error: {str(e)}</div>', unsafe_allow_html=True)
    finally:
        os.unlink(tmp_path)

elif not st.session_state.MODEL_DIR or not st.session_state.FILTER_DIR:
    st.markdown("""
    <div class="error-box">
    ❌ Required resources could not be downloaded.<br><br>
    Possible solutions:
    <ol>
        <li>Click the 'Retry Download Resources' button in the sidebar</li>
        <li>Check your internet connection</li>
        <li>Try again later</li>
    </ol>
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
4. Compare filter performance

**Supported formats:**
- Text files (.txt, .dat)
- FITS files (.fits)
- Spectrum files (.spec)

**Note:** First-time setup may take a few minutes to download all required resources.
""")
