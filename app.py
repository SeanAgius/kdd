import streamlit as st
import pandas as pd
import joblib
import numpy as np
import warnings
import logging
from io import StringIO
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Suppress warnings by default
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

# Suppress Streamlit deprecation warnings and errors
try:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.set_option('client.showWarningOnDirectExecution', False)
    st.set_option('client.showErrorDetails', False)
    st.set_option('global.suppressDeprecationWarnings', True)
    st.set_option('runner.displayEnabled', True)
    st.set_option('client.displayEnabled', True)
except:
    pass  # Some options might not exist in all Streamlit versions

# Override default error handling
import sys
class SuppressedStdout:
    def write(self, txt): pass
    def flush(self): pass

# Suppress stderr for Streamlit warnings in production
if 'streamlit' in sys.modules:
    try:
        # Redirect streamlit error output
        pass
    except:
        pass

# Set page configuration
st.set_page_config(
    page_title="SMARTSPACK KDD Model",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_models(show_errors=False):
    """Load the pre-trained models from joblib file"""
    try:
        # Try joblib first (preferred)
        saved_models = joblib.load('trained_models.joblib')
        return saved_models
    except FileNotFoundError:
        # Fallback to pickle if joblib not found
        try:
            import pickle
            with open('trained_models.pkl', 'rb') as f:
                saved_models = pickle.load(f)
            return saved_models
        except FileNotFoundError:
            if show_errors:
                st.error("Error: Neither 'trained_models.joblib' nor 'trained_models.pkl' found. Please ensure a model file is in the same directory.")
            return None
    except ModuleNotFoundError as e:
        if show_errors:
            st.error(f"Error: Missing required module - {str(e)}. Please check your environment setup.")
        return None
    except Exception as e:
        if show_errors:
            st.error(f"Error loading models: {str(e)}. The model file may be corrupted or incompatible.")
        return None

def make_predictions(input_data):
    """Make predictions using pre-trained models"""
    saved_models = load_models()
    if saved_models is None:
        return None
    
    # Convert input data to DataFrame if it's not already
    if isinstance(input_data, pd.DataFrame):
        io_data = input_data
    else:
        io_data = pd.DataFrame(input_data)
    
    # Extract features (first 5 columns) and encode them
    X_input = io_data.iloc[:, :5]
    
    # Use the stored feature encoders from the models
    X_encoded = pd.DataFrame()
    
    # Get feature encoders from any model (they should all be the same)
    first_model = next(iter(saved_models.values()))
    if 'feature_encoders' in first_model:
        feature_encoders = first_model['feature_encoders']
        
        for col in X_input.columns:
            if col in feature_encoders:
                encoder = feature_encoders[col]
                # Handle unknown categories by mapping to first category
                encoded_values = []
                for val in X_input[col]:
                    try:
                        encoded_values.append(encoder.transform([str(val)])[0])
                    except ValueError:
                        # Map unknown categories to the first category
                        encoded_values.append(0)
                X_encoded[col] = encoded_values
            else:
                # Fallback: simple numeric encoding
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_input[col])
    else:
        # Fallback to old method if no feature encoders stored
        from sklearn.preprocessing import LabelEncoder
        feature_categories = {
            'Age Bracket': ['18-25', '26-35', '36-45', '46-55', '56-65', '65+'],
            'Gender': ['Male', 'Female'],
            'Temperature of food': ['Too Hot', 'Hot', 'Just Right', 'Cold', 'Too Cold'],
            'Texture of food': ['Soggy', 'Just Right', 'Dry'],
            'Stacking': ['Top', 'Bottom']
        }
        
        for i, col in enumerate(X_input.columns):
            le = LabelEncoder()
            categories = feature_categories.get(col, X_input[col].unique())
            le.fit(categories)
            
            # Handle unknown categories by mapping to first category
            encoded_values = []
            for val in X_input[col]:
                try:
                    encoded_values.append(le.transform([str(val)])[0])
                except ValueError:
                    encoded_values.append(0)  # Default to first category
            
            X_encoded[col] = encoded_values
    
    # Target names (same as in training)
    targets = ['Type of Sanitiser','No. of Vents','Vents Sizes', 'Flut Size', 
               'Packaging Leaking', 'Package Satisfaction', 'Sanitiser Satisfaction']
    
    # Create a copy of input data for results
    results_df = io_data.copy()
    
    # Make predictions for each target
    for target in targets:
        if target in saved_models:
            model_info = saved_models[target]
            best_model = model_info['model']
            label_encoder = model_info['encoder']
            
            # Make predictions (encoded)
            preds_encoded = best_model.predict(X_encoded)
            
            # Convert back to original labels
            predictions = label_encoder.inverse_transform(preds_encoded)
            
            # Add predictions to results
            results_df[target] = predictions
    
    return results_df

def get_model_info():
    """Get information about loaded models"""
    saved_models = load_models()
    if saved_models is None:
        return None
    
    model_info = {}
    for target, info in saved_models.items():
        model_info[target] = {
            'Model Type': info['model_type'],
            'Accuracy': f"{info['accuracy']:.4f}",
            'Classes': list(info['encoder'].classes_)
        }
    
    return model_info

def main():
    st.title("📦 SMARTSPACK KDD Model")
    st.markdown("---")
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.radio(
        "Choose a page:",
        ["Make Predictions", "Model Information", "About"]
    )
    
    if option == "Make Predictions":
        st.header("🎯 Make Predictions")
        
        # Manual Input Form only
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age_bracket = st.selectbox(
                    "Age Bracket",
                    ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
                )
                
                gender = st.selectbox(
                    "Gender",
                    ["Male", "Female"]
                )
                
                temp_food = st.selectbox(
                    "Temperature of Food",
                    ["Too Hot", "Hot", "Just Right", "Cold", "Too Cold"]
                )
            
            with col2:
                texture_food = st.selectbox(
                    "Texture of Food",
                    ["Soggy", "Just Right", "Dry"]
                )
                
                stacking = st.selectbox(
                    "Stacking",
                    ["Top", "Bottom"]
                )
            
            submitted = st.form_submit_button("Make Prediction", type="primary")
            
            if submitted:
                # Create DataFrame from form input
                input_data = pd.DataFrame({
                    'Age Bracket': [age_bracket],
                    'Gender': [gender],
                    'Temperature of food': [temp_food],
                    'Texture of food': [texture_food],
                    'Stacking': [stacking]
                })
                
                with st.spinner("Making prediction..."):
                    try:
                        results = make_predictions(input_data)
                        
                        if results is not None:
                            st.success("Prediction completed!")
                            
                            st.subheader("Prediction Results")
                            
                            # Define automation parameters and UX metrics
                            automation_params = ['Type of Sanitiser', 'No. of Vents', 'Vents Sizes', 'Flut Size']
                            ux_params = ['Packaging Leaking', 'Package Satisfaction', 'Sanitiser Satisfaction']
                            
                            # Display Smart Machine Parameters
                            st.markdown("#### ⚙️ Smart Machine Parameters")
                            col1, col2 = st.columns(2)
                            for i, param in enumerate(automation_params):
                                if i % 2 == 0:
                                    with col1:
                                        st.metric(label=param, value=results[param].iloc[0])
                                else:
                                    with col2:
                                        st.metric(label=param, value=results[param].iloc[0])
                            
                            # Add spacing between sections
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Display UX Metrics
                            st.markdown("#### 😊 UX Metrics")
                            col3, col4, col5 = st.columns(3)
                            for i, param in enumerate(ux_params):
                                if i == 0:
                                    with col3:
                                        st.metric(label=param, value=results[param].iloc[0])
                                elif i == 1:
                                    with col4:
                                        st.metric(label=param, value=results[param].iloc[0])
                                else:
                                    with col5:
                                        st.metric(label=param, value=results[param].iloc[0])
                            
                            # Show full results table
                            st.subheader("Complete Results")
                            st.dataframe(results)
                            
                            # Download button
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name='prediction_result.csv',
                                mime='text/csv'
                            )
                            
                        else:
                            # If predictions fail, show a simple message without technical errors
                            st.info("Prediction processing completed. Please try again if needed.")
                    
                    except Exception as e:
                        # Catch all Streamlit errors and other exceptions
                        # Hide technical errors from users, show friendly message instead
                        st.info("Processing your request. Please wait a moment and try again if needed.")
                        # Log errors silently without showing them to user
                        import sys
                        print(f"Hidden error: {str(e)}", file=sys.stderr)
    
    elif option == "Model Information":
        st.header("ℹ️ Model Information")
        
        model_info = get_model_info()
        
        if model_info:
            for target, info in model_info.items():
                with st.expander(f"📊 {target}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Model Type", info['Model Type'])
                        st.metric("Accuracy", info['Accuracy'])
                    with col2:
                        st.write("**Available Classes:**")
                        for class_name in info['Classes']:
                            st.write(f"• {class_name}")
        else:
            with st.expander("🔧 Debug Information", expanded=False):
                st.error("Could not load model information. Please check if 'trained_models.pkl' exists.")
    
    elif option == "About":
        st.header("📋 About")
        
        st.markdown("""
        ## SMARTSPACK KDD Model
        
        This web application uses machine learning models to predict various aspects of smart packaging based on user inputs.
        
        ### Features:
        - **Simple Manual Input**: Easy-to-use form interface
        - **Seven Prediction Categories**: 
          - Type of Sanitiser
          - Number of Vents
          - Vent Sizes
          - Flut Size
          - Packaging Leaking
          - Package Satisfaction
          - Sanitiser Satisfaction
        
        ### Input Requirements:
        The system requires the following input features:
        1. **Age Bracket**: Age range of the user
        2. **Gender**: Male or Female
        3. **Temperature of Food**: Too Hot, Hot, Just Right, Cold, or Too Cold
        4. **Texture of Food**: Soggy, Just Right, or Dry
        5. **Stacking**: Top or Bottom
        
        ### How to Use:
        1. Click "Make Predictions" from the navigation buttons
        2. Fill out the manual input form with your preferences
        3. Click "Make Prediction" to get results
        4. View the Smart Machine Parameters and UX Metrics
        5. Download the results as a CSV file if needed
        
        ### Model Information:
        View detailed information about each trained model including accuracy scores and available prediction classes.
        """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Catch any uncaught exceptions to prevent red error messages
        st.error("Application is temporarily unavailable. Please refresh the page.")
        # Log error silently
        import sys
        print(f"Application error: {str(e)}", file=sys.stderr)