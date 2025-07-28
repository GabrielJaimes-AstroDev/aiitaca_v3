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
# INITIALIZE SESSION STATE
# =============================================
if 'resources_downloaded' not in st.session_state:
    st.session_state.resources_downloaded = False
    st.session_state.MODEL_DIR = None
    st.session_state.FILTER_DIR = None
    st.session_state.downloaded_files = {'models': [], 'filters': []}
    st.session_state.initial_setup_complete = False

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
    """Load content from external text file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        st.error(f"Error loading file {filename}: {str(e)}")
        return ""

def load_urls_config(filename):
    """Load configuration URLs with robust validation"""
    urls = {}
    required_keys = ['MODEL_FOLDER_URL', 'FILTER_FOLDER_URL']
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    urls[key.strip()] = value.strip()
    except Exception as e:
        st.error(f"Error reading {filename}: {str(e)}")
    
    missing_keys = [key for key in required_keys if key not in urls]
    if missing_keys:
        st.error(f"Missing required URLs in {filename}: {', '.join(missing_keys)}")
    
    return urls

# Load external resources
CSS_STYLES = load_external_file('styles_css.txt')
INITIAL_DESCRIPTION = load_external_file('description.txt')
urls = load_urls_config('urls.txt')
MODEL_FOLDER_URL = urls.get('MODEL_FOLDER_URL')
FILTER_FOLDER_URL = urls.get('FILTER_FOLDER_URL')

# Critical validation
if not MODEL_FOLDER_URL or not FILTER_FOLDER_URL:
    st.error("""
    ❌ Essential configuration missing. Application cannot continue.
    Please verify urls.txt contains:
    MODEL_FOLDER_URL=your_url_here
    FILTER_FOLDER_URL=your_url_here
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
    """List all downloaded files with detailed information"""
    file_list = []
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                if not file.startswith('.'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, directory)
                    try:
                        size = os.path.getsize(full_path)
                        file_list.append({
                            'path': rel_path,
                            'size': f"{size/1024:.1f} KB",
                            'full_path': full_path,
                            'parent_dir': os.path.basename(root)
                        })
                    except Exception:
                        file_list.append({
                            'path': rel_path,
                            'size': 'Error',
                            'full_path': full_path,
                            'parent_dir': os.path.basename(root)
                        })
    except Exception as e:
        st.error(f"Error listing files: {str(e)}")
    return file_list

def download_google_drive_folder(folder_url, output_dir):
    """Download content from Google Drive folder"""
    try:
        folder_id = folder_url.split('folders/')[-1].split('?')[0]
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
        
        gdown.download_folder(
            id=folder_id,
            output=output_dir,
            quiet=True,
            use_cookies=False
        )
        return True
    except Exception as e:
        st.error(f"Download error: {str(e)}")
        return False

def download_resources():
    """Download all required resources"""
    with st.spinner("🔽 Downloading models..."):
        if not download_google_drive_folder(MODEL_FOLDER_URL, "downloaded_models"):
            return None, None
    
    with st.spinner("🔽 Downloading filters..."):
        if not download_google_drive_folder(FILTER_FOLDER_URL, "downloaded_filters"):
            return None, None
    
    return "downloaded_models", "downloaded_filters"

def robust_read_file(file_path):
    """Read spectrum or filter files with robust handling"""
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
                lines = [line.replace(',', '.').strip() 
                         for line in decoded.splitlines() 
                         if line.strip() and not line.strip().startswith(('!', '//', '#'))]
                
                if not lines:
                    continue
                
                data = np.genfromtxt(lines)
                if data.ndim == 2 and data.shape[1] >= 2:
                    return data[:, 0], data[:, 1]
            except (UnicodeDecodeError, ValueError):
                continue
        
        raise ValueError("Could not read file with standard encodings")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
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
        filter_intensity = filter_intensity / np.max(filter_intensity)
        
        # Create mask
        significant_mask = filter_intensity > 0.01
        
        # Validate input spectrum
        valid_points = (~np.isnan(spectrum_intensity)) & (~np.isinf(spectrum_intensity))
        if np.sum(valid_points) < 2:
            raise ValueError("Insufficient valid points in spectrum")
        
        # Interpolate
        interp_func = interp1d(
            spectrum_freq[valid_points],
            spectrum_intensity[valid_points],
            kind='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # Apply filter
        filtered_intensity = interp_func(filter_freq) * filter_intensity
        
        # Create full output
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
        st.error(f"Filter error: {str(e)}")
        return None

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
# RESOURCE DOWNLOAD AND INITIALIZATION
# =============================================
if not st.session_state.initial_setup_complete:
    if not st.session_state.resources_downloaded:
        with st.spinner("⚙️ Initializing application resources..."):
            st.session_state.MODEL_DIR, st.session_state.FILTER_DIR = download_resources()
            
            if st.session_state.MODEL_DIR and st.session_state.FILTER_DIR:
                try:
                    st.session_state.downloaded_files['models'] = list_downloaded_files(st.session_state.MODEL_DIR)
                    st.session_state.downloaded_files['filters'] = list_downloaded_files(st.session_state.FILTER_DIR)
                    st.session_state.resources_downloaded = True
                    st.session_state.initial_setup_complete = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Initialization error: {str(e)}")
                    st.session_state.resources_downloaded = False
    else:
        st.session_state.initial_setup_complete = True

# =============================================
# SIDEBAR - CONFIGURATION
# =============================================
st.sidebar.title("Configuration")

with st.sidebar:
    with st.expander("📁 Downloaded Resources", expanded=False):
        if st.session_state.MODEL_DIR:
            st.subheader("Models")
            st.code(f"./{st.session_state.MODEL_DIR}")
            
        if st.session_state.FILTER_DIR:
            st.subheader("Filters")
            st.code(f"./{st.session_state.FILTER_DIR}")

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
    help="Upload your spectrum file"
)

# =============================================
# FILTER PROCESSING AND VISUALIZATION
# =============================================
if input_file is not None and st.session_state.resources_downloaded:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(input_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Read input spectrum
        input_freq, input_spec = robust_read_file(tmp_path)
        if input_freq is None:
            raise ValueError("Could not read spectrum file")
        
        # Convert to GHz if needed
        if np.mean(input_freq) > 1e6:
            input_freq = input_freq / 1e9
        
        # Get all filter files
        filter_files = []
        for root, _, files in os.walk(st.session_state.FILTER_DIR):
            for file in files:
                if file.endswith('.txt'):
                    filter_files.append(os.path.join(root, file))
        
        if not filter_files:
            raise ValueError("No filter files found")
        
        # Process with all filters
        filtered_results = []
        failed_filters = []
        
        with st.spinner("🔍 Applying filters..."):
            progress_bar = st.progress(0)
            for i, filter_file in enumerate(filter_files):
                progress_bar.progress((i + 1) / len(filter_files))
                result = apply_spectral_filter(input_freq, input_spec, filter_file)
                if result:
                    # Create temporary output file
                    output_filename = f"filtered_{result['filter_name']}.txt"
                    output_path = os.path.join(tempfile.gettempdir(), output_filename)
                    
                    header = f"Frequency(GHz)\tIntensity(K)\n# Filter applied: {result['filter_name']}"
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
        
        if not filtered_results:
            raise ValueError(f"No filters applied successfully. {len(failed_filters)} failed.")
        
        st.markdown(f'<div class="success-box">✅ Applied {len(filtered_results)} filters</div>', unsafe_allow_html=True)
        
        if failed_filters:
            st.markdown(f'<div class="warning-box">⚠ Failed filters: {", ".join(failed_filters)}</div>', unsafe_allow_html=True)
        
        # Visualization tabs
        tab1, tab2 = st.tabs(["Interactive Spectrum", "Filter Details"])
        
        with tab1:
            fig_main = go.Figure()
            fig_main.add_trace(go.Scatter(
                x=input_freq,
                y=input_spec,
                mode='lines',
                name='Original',
                line=dict(color='white', width=2)
            ))
            
            colors = ['#1E88E5', '#FF5722', '#4CAF50', '#FFC107', '#9C27B0']
            for i, result in enumerate(filtered_results):
                fig_main.add_trace(go.Scatter(
                    x=result['filtered_data']['freq'],
                    y=result['filtered_data']['intensity'],
                    mode='lines',
                    name=result['name'],
                    line=dict(color=colors[i % len(colors)], width=1.5)
                ))
            
            fig_main.update_layout(
                title="Spectrum Filtering Results",
                xaxis_title="Frequency (GHz)",
                yaxis_title="Intensity (K)",
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
        
        with tab2:
            for result in filtered_results:
                with st.expander(f"🔍 {result['name']} (from {result['filtered_data']['parent_dir']})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_filter = go.Figure()
                        fig_filter.add_trace(go.Scatter(
                            x=result['filtered_data']['freq'],
                            y=result['filtered_data']['filter_profile'],
                            line=dict(color='#1E88E5')
                        ))
                        fig_filter.update_layout(
                            title="Filter Profile",
                            height=300,
                            plot_bgcolor='#0D0F14',
                            paper_bgcolor='#0D0F14'
                        )
                        st.plotly_chart(fig_filter, use_container_width=True)
                    
                    with col2:
                        fig_compare = go.Figure()
                        fig_compare.add_trace(go.Scatter(
                            x=result['original_freq'],
                            y=result['original_intensity'],
                            line=dict(color='white')
                        ))
                        fig_compare.add_trace(go.Scatter(
                            x=result['filtered_data']['freq'],
                            y=result['filtered_data']['intensity'],
                            line=dict(color='#FF5722')
                        ))
                        fig_compare.update_layout(
                            title="Comparison",
                            height=300,
                            plot_bgcolor='#0D0F14',
                            paper_bgcolor='#0D0F14'
                        )
                        st.plotly_chart(fig_compare, use_container_width=True)
                    
                    # Download button
                    with open(result['output_path'], 'rb') as f:
                        st.download_button(
                            label=f"📥 Download {result['name']} filtered spectrum",
                            data=f,
                            file_name=os.path.basename(result['output_path']),
                            mime='text/plain',
                            key=f"download_{result['name']}",
                            use_container_width=True
                        )
    
    except Exception as e:
        st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
    finally:
        os.unlink(tmp_path)

elif not st.session_state.resources_downloaded:
    st.markdown("""
    <div class="error-box">
    ❌ Resources not downloaded.
    <ol>
        <li>Click 'Retry Download Resources'</li>
        <li>Check internet connection</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("ℹ️ Upload a spectrum file to begin")

# =============================================
# INSTRUCTIONS
# =============================================
st.sidebar.markdown("""
**Instructions:**
1. Upload spectrum file
2. System applies all filters
3. View results in tabs

**Formats:**
- Text (.txt, .dat)
- FITS (.fits)
- Spectrum (.spec)
""")
