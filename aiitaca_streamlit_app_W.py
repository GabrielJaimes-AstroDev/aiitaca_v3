def load_prediction_models(model_dir):
    """Load the machine learning models for prediction with version compatibility handling"""
    try:
        # Find model files
        model_files = []
        for root, _, files in os.walk(model_dir):
            for file in files:
                if file.endswith('.pkl') and 'random_forest' in file.lower():
                    model_files.append(os.path.join(root, file))
        
        if len(model_files) < 4:
            st.error("Not all required model files found")
            return None, None, None, None, None
        
        # Load with compatibility warning suppression
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            rf_tex = None
            rf_logn = None
            x_scaler = None
            tex_scaler = None
            logn_scaler = None
            
            for model_file in model_files:
                try:
                    if 'random_forest_tex' in model_file.lower():
                        rf_tex = joblib.load(model_file)
                    elif 'random_forest_logn' in model_file.lower():
                        rf_logn = joblib.load(model_file)
                    elif 'x_scaler' in model_file.lower():
                        x_scaler = joblib.load(model_file)
                    elif 'tex_scaler' in model_file.lower():
                        tex_scaler = joblib.load(model_file)
                    elif 'logn_scaler' in model_file.lower():
                        logn_scaler = joblib.load(model_file)
                except Exception as e:
                    st.error(f"Error loading {model_file}: {str(e)}")
                    continue
        
        if all([rf_tex, rf_logn, x_scaler, tex_scaler, logn_scaler]):
            return rf_tex, rf_logn, x_scaler, tex_scaler, logn_scaler
        else:
            st.error("Some model components could not be loaded")
            return None, None, None, None, None
            
    except Exception as e:
        st.error(f"Error loading prediction models: {str(e)}")
        return None, None, None, None, None
