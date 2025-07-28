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
import pandas as pd
import joblib
from io import StringIO
import warnings

# =============================================
# INITIALIZE SESSION STATE
# =============================================
if not hasattr(st.session_state, 'resources_downloaded'):
    st.session_state.resources_downloaded = False
    st.session_state.MODEL_DIR = None
    st.session_state.FILTER_DIR = None
    st.session_state.downloaded_files = {'models': [], 'filters': []}
    st.session_state.prediction_models_loaded = False
    st.session_state.prediction_models = None

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
with open('styles.txt', 'r') as f:
    css_styles = f.read()
st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)

# =============================================
# HELPER FUNCTIONS
# =============================================
def list_downloaded_files(directory):
    """Recursively list all downloaded files with detailed information"""
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
    # Load URLs from external file
    with open('urls.txt', 'r') as f:
        urls = f.read().splitlines()
    
    MODEL_FOLDER_URL = urls[0]  # First line is models URL
    FILTER_FOLDER_URL = urls[1]  # Second line is filters URL
    
    MODEL_DIR = "downloaded_models"
    FILTER_DIR = "downloaded_filters"
    
    # Download models
    with st.spinner("🔽 Downloading models (this may take several minutes)..."):
        if not download_google_drive_folder(MODEL_FOLDER_URL, MODEL_DIR):
            return None, None
    
    # Download filters
    with st.spinner("🔽 Downloading filters..."):
        if not download_google_drive_folder(FILTER_FOLDER_URL, FILTER_DIR):
            return None, None
    
    return MODEL_DIR, FILTER_DIR

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

def load_prediction_models(models_dir):
    """Load the prediction models from the downloaded resources"""
    try:
        # Find model files
        model_files = []
        for root, _, files in os.walk(models_dir):
            for file in files:
                if file.endswith('.pkl'):
                    model_files.append(os.path.join(root, file))
        
        if not model_files:
            st.error("No se encontraron archivos de modelo (.pkl) en el directorio de modelos descargados")
            return None
        
        # Load models
        models = {}
        for model_file in model_files:
            model_name = os.path.basename(model_file)
            if 'random_forest_tex' in model_name:
                models['rf_tex'] = joblib.load(model_file)
            elif 'random_forest_logn' in model_name:
                models['rf_logn'] = joblib.load(model_file)
            elif 'x_scaler' in model_name:
                models['x_scaler'] = joblib.load(model_file)
            elif 'tex_scaler' in model_name:
                models['tex_scaler'] = joblib.load(model_file)
            elif 'logn_scaler' in model_name:
                models['logn_scaler'] = joblib.load(model_file)
        
        # Verify all required models are loaded
        required_models = ['rf_tex', 'rf_logn', 'x_scaler', 'tex_scaler', 'logn_scaler']
        if not all(m in models for m in required_models):
            missing = [m for m in required_models if m not in models]
            st.error(f"Faltan modelos requeridos: {', '.join(missing)}")
            return None
        
        return models
    except Exception as e:
        st.error(f"Error cargando modelos de predicción: {str(e)}")
        return None

def process_spectrum_for_prediction(freq, intensity, interpolation_length=64610, min_required_points=1000):
    """Process spectrum for prediction"""
    try:
        # Create a DataFrame from the input
        data = pd.DataFrame({
            'freq': freq,
            'intensity': intensity
        })
        
        # Clean data
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(data) < min_required_points:
            st.error(f"Espectro tiene pocos puntos válidos ({len(data)})")
            return None
        
        # Normalize frequency
        min_freq = data['freq'].min()
        max_freq = data['freq'].max()
        freq_range = max_freq - min_freq
        if freq_range == 0:
            st.error("Rango de frecuencia cero en el espectro")
            return None
        
        normalized_freq = (data['freq'] - min_freq) / freq_range
        
        # Interpolate
        interp_func = interp1d(normalized_freq, data['intensity'], 
                              kind='linear', bounds_error=False, 
                              fill_value=(data['intensity'].iloc[0], data['intensity'].iloc[-1]))
        new_freq = np.linspace(0, 1, interpolation_length)
        interpolated_intensity = interp_func(new_freq).astype(np.float32)
        
        # Handle any remaining NaNs
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
        st.error(f"Error procesando espectro para predicción: {str(e)}")
        return None

def make_predictions(spectrum_data, models):
    """Make predictions using loaded models"""
    try:
        # Scale the spectrum
        scaled_spectrum = models['x_scaler'].transform([spectrum_data])
        
        # Make predictions
        tex_pred_scaled = models['rf_tex'].predict(scaled_spectrum)
        tex_pred = models['tex_scaler'].inverse_transform(tex_pred_scaled.reshape(-1, 1))[0,0]
        
        logn_pred_scaled = models['rf_logn'].predict(scaled_spectrum)
        logn_pred = models['logn_scaler'].inverse_transform(logn_pred_scaled.reshape(-1, 1))[0,0]
        
        return tex_pred, logn_pred
    except Exception as e:
        st.error(f"Error realizando predicciones: {str(e)}")
        return None, None

def plot_prediction_results(tex_pred, logn_pred):
    """Create plotly figure for prediction results"""
    fig = go.Figure()
    
    # LogN plot
    fig.add_trace(go.Scatter(
        x=[18.1857],
        y=[logn_pred],
        mode='markers',
        marker=dict(
            color='red',
            size=15,
            line=dict(width=2, color='black')
        ),
        name='LogN predicho',
        text=[f"Pred: {logn_pred:.2f}"],
        hoverinfo='text',
        xaxis='x1',
        yaxis='y1'
    ))
    
    # Tex plot
    fig.add_trace(go.Scatter(
        x=[203.492],
        y=[tex_pred],
        mode='markers',
        marker=dict(
            color='red',
            size=15,
            line=dict(width=2, color='black')
        ),
        name='Tex predicho',
        text=[f"Pred: {tex_pred:.1f} K"],
        hoverinfo='text',
        xaxis='x2',
        yaxis='y2'
    ))
    
    # Update layout for subplots
    fig.update_layout(
        title='Resultados de Predicción',
        grid=dict(rows=1, columns=2, pattern='independent'),
        plot_bgcolor='#0D0F14',
        paper_bgcolor='#0D0F14',
        font=dict(color='white'),
        height=400
    )
    
    # Update xaxis properties for LogN plot
    fig.update_xaxes(title_text='LogN de referencia', row=1, col=1)
    fig.update_yaxes(title_text='LogN predicho', row=1, col=1)
    
    # Update xaxis properties for Tex plot
    fig.update_xaxes(title_text='Tex de referencia (K)', row=1, col=2)
    fig.update_yaxes(title_text='Tex predicho (K)', row=1, col=2)
    
    return fig

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

# Load description from external file
with open('description.txt', 'r') as f:
    description_content = f.read()
st.markdown(f"<div class='description-panel'>{description_content}</div>", unsafe_allow_html=True)

# =============================================
# MAIN INTERFACE
# =============================================
# Download resources on startup
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

# Show downloaded resources
with st.sidebar:
    st.header("📁 Downloaded Resources")
    
    # Models section
    if st.session_state.MODEL_DIR and os.path.exists(st.session_state.MODEL_DIR):
        st.subheader("Models Directory")
        st.code(f"./{st.session_state.MODEL_DIR}", language="bash")
        
        # Display model files in a compact way
        if st.session_state.downloaded_files['models']:
            st.markdown("**Model files:**")
            model_files_text = "\n".join(
                f"- {file['path']} ({file['size']})" 
                for file in st.session_state.downloaded_files['models']
            )
            st.text_area("Model files list", value=model_files_text, height=150, label_visibility="collapsed")
        else:
            st.warning("No model files found")
    
    # Filters section
    if st.session_state.FILTER_DIR and os.path.exists(st.session_state.FILTER_DIR):
        st.subheader("Filters Directory")
        st.code(f"./{st.session_state.FILTER_DIR}", language="bash")
        
        # Display filter files in a compact way
        if st.session_state.downloaded_files['filters']:
            st.markdown("**Filter files:**")
            filter_files_text = "\n".join(
                f"- {file['path']} ({file['size']})" 
                for file in st.session_state.downloaded_files['filters']
            )
            st.text_area("Filter files list", value=filter_files_text, height=150, label_visibility="collapsed")
        else:
            st.warning("No filter files found")

    # Button to retry download
    if st.button("🔄 Retry Download Resources"):
        st.session_state.MODEL_DIR, st.session_state.FILTER_DIR = download_resources()
        if st.session_state.MODEL_DIR and st.session_state.FILTER_DIR:
            st.session_state.downloaded_files['models'] = list_downloaded_files(st.session_state.MODEL_DIR)
            st.session_state.downloaded_files['filters'] = list_downloaded_files(st.session_state.FILTER_DIR)
            st.session_state.resources_downloaded = True
            st.rerun()

# File selector
input_file = st.sidebar.file_uploader(
    "Input Spectrum File",
    type=['.txt', '.dat', '.fits', '.spec'],
    help="Upload your spectrum file (TXT, DAT, FITS, SPEC)"
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
        tab1, tab2, tab3 = st.tabs(["Interactive Spectrum", "Filter Details", "Molecular Parameters Prediction"])
        
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
                            showlegend=False
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
                            line=dict(color='#FF5722', width=1))
                        )
                        fig_compare.update_layout(
                            title="Original vs Filtered",
                            height=300,
                            plot_bgcolor='#0D0F14',
                            paper_bgcolor='#0D0F14',
                            showlegend=False
                        )
                        st.plotly_chart(fig_compare, use_container_width=True)
                    
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
            st.markdown("## Molecular Parameters Prediction")
            st.markdown("Predict LogN (column density) and Tex (excitation temperature) using machine learning models.")
            
            # Load prediction models if not already loaded
            if not st.session_state.prediction_models_loaded:
                with st.spinner("Loading prediction models..."):
                    st.session_state.prediction_models = load_prediction_models(st.session_state.MODEL_DIR)
                    st.session_state.prediction_models_loaded = True
            
            if st.session_state.prediction_models:
                # Select which filtered spectrum to use for prediction
                filter_options = [f"{res['name']} (from {res['filtered_data']['parent_dir']})" for res in filtered_results]
                selected_filter = st.selectbox(
                    "Select filtered spectrum for prediction:",
                    options=filter_options,
                    index=0
                )
                
                selected_index = filter_options.index(selected_filter)
                selected_result = filtered_results[selected_index]
                
                # Process spectrum for prediction
                with st.spinner("Processing spectrum for prediction..."):
                    processed_spectrum = process_spectrum_for_prediction(
                        selected_result['filtered_data']['freq'],
                        selected_result['filtered_data']['intensity']
                    )
                
                if processed_spectrum is not None:
                    # Make predictions
                    with st.spinner("Making predictions..."):
                        tex_pred, logn_pred = make_predictions(processed_spectrum, st.session_state.prediction_models)
                    
                    if tex_pred is not None and logn_pred is not None:
                        st.success("Prediction completed successfully!")
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(label="Predicted Excitation Temperature (Tex)", value=f"{tex_pred:.2f} K")
                        with col2:
                            st.metric(label="Predicted Column Density (LogN)", value=f"{logn_pred:.2f}")
                        
                        # Plot results
                        st.plotly_chart(
                            plot_prediction_results(tex_pred, logn_pred),
                            use_container_width=True
                        )
                        
                        # Show details
                        with st.expander("Prediction Details"):
                            st.markdown(f"""
                            **Filter used:** {selected_result['name']}  
                            **Source directory:** {selected_result['filtered_data']['parent_dir']}  
                            **Number of points in spectrum:** {len(selected_result['filtered_data']['freq'])}  
                            **Intensity range:** {np.min(selected_result['filtered_data']['intensity']):.2e} to {np.max(selected_result['filtered_data']['intensity']):.2e} K
                            """)
                    else:
                        st.error("Failed to make predictions")
                else:
                    st.error("Failed to process spectrum for prediction")
            else:
                st.error("Prediction models could not be loaded")
    
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
4. Download filtered spectra as needed
5. Predict molecular parameters in the Prediction tab

**Supported formats:**
- Text files (.txt, .dat)
- FITS files (.fits)
- Spectrum files (.spec)

**Note:** First-time setup may take a few minutes to download all required resources.
""")
