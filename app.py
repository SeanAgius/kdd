import streamlit as st
import pandas as pd
import joblib
import numpy as np
from io import StringIO
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Set page configuration
st.set_page_config(
    page_title="Smart Pack Prediction System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_models():
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
            st.error("Error: Neither 'trained_models.joblib' nor 'trained_models.pkl' found. Please ensure a model file is in the same directory.")
            return None
    except ModuleNotFoundError as e:
        st.error(f"Error: Missing required module - {str(e)}. Please check your environment setup.")
        return None
    except Exception as e:
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
            'Gender': ['Male', 'Female', 'Other'],
            'Temperature of food': ['Too Cold', 'Just Right', 'Too Hot'],
            'Texture of food': ['Too Soft', 'Just Right', 'Too Hard', 'Dry'],
            'Stacking': ['Top', 'Middle', 'Bottom']
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
    st.title("📦 Smart Pack Prediction System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    option = st.sidebar.selectbox(
        "Choose an option:",
        ["Make Predictions", "Model Information", "About"]
    )
    
    if option == "Make Predictions":
        st.header("🎯 Make Predictions")
        
        # Option to choose input method
        input_method = st.radio(
            "Choose input method:",
            ["Upload CSV File", "Manual Input Form"]
        )
        
        if input_method == "Upload CSV File":
            st.subheader("Upload CSV File")
            uploaded_file = st.file_uploader(
                "Choose a CSV file", 
                type="csv",
                help="Upload a CSV file with the following columns: Age Bracket, Gender, Temperature of food, Texture of food, Stacking"
            )
            
            if uploaded_file is not None:
                try:
                    # Read the uploaded file
                    df = pd.read_csv(uploaded_file)
                    
                    st.subheader("Input Data Preview")
                    st.dataframe(df)
                    
                    # Make predictions
                    if st.button("Make Predictions", type="primary"):
                        with st.spinner("Making predictions..."):
                            results = make_predictions(df)
                            
                            if results is not None:
                                st.success("Predictions completed!")
                                
                                st.subheader("Prediction Results")
                                
                                # Define automation parameters and UX metrics
                                automation_params = ['Type of Sanitiser', 'No. of Vents', 'Vents Sizes', 'Flut Size']
                                ux_params = ['Packaging Leaking', 'Package Satisfaction', 'Sanitiser Satisfaction']
                                
                                # Display summary for first row if available
                                if len(results) > 0:
                                    st.markdown("#### 🤖 Automation Parameters (First Row)")
                                    col1, col2 = st.columns(2)
                                    for i, param in enumerate(automation_params):
                                        if i % 2 == 0:
                                            with col1:
                                                st.metric(label=param, value=results[param].iloc[0])
                                        else:
                                            with col2:
                                                st.metric(label=param, value=results[param].iloc[0])
                                    
                                    st.markdown("#### 📱 UX Metrics (First Row)")
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
                                
                                st.markdown("#### 📊 Complete Results Table")
                                st.dataframe(results)
                                
                                # Download button
                                csv = results.to_csv(index=False)
                                st.download_button(
                                    label="Download Results as CSV",
                                    data=csv,
                                    file_name='predictions_output.csv',
                                    mime='text/csv'
                                )
                            else:
                                st.error("Failed to make predictions. Please check your model file.")
                
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        else:  # Manual Input Form
            st.subheader("Manual Input Form")
            
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    age_bracket = st.selectbox(
                        "Age Bracket",
                        ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
                    )
                    
                    gender = st.selectbox(
                        "Gender",
                        ["Male", "Female", "Other"]
                    )
                    
                    temp_food = st.selectbox(
                        "Temperature of Food",
                        ["Too Cold", "Just Right", "Too Hot"]
                    )
                
                with col2:
                    texture_food = st.selectbox(
                        "Texture of Food",
                        ["Too Soft", "Just Right", "Too Hard", "Dry"]
                    )
                    
                    stacking = st.selectbox(
                        "Stacking",
                        ["Top", "Middle", "Bottom"]
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
                        results = make_predictions(input_data)
                        
                        if results is not None:
                            st.success("Prediction completed!")
                            
                            st.subheader("Prediction Results")
                            
                            # Define automation parameters and UX metrics
                            automation_params = ['Type of Sanitiser', 'No. of Vents', 'Vents Sizes', 'Flut Size']
                            ux_params = ['Packaging Leaking', 'Package Satisfaction', 'Sanitiser Satisfaction']
                            
                            # Display Automation Parameters
                            st.markdown("#### 🤖 Automation Parameters")
                            col1, col2 = st.columns(2)
                            for i, param in enumerate(automation_params):
                                if i % 2 == 0:
                                    with col1:
                                        st.metric(label=param, value=results[param].iloc[0])
                                else:
                                    with col2:
                                        st.metric(label=param, value=results[param].iloc[0])
                            
                            # Display UX Metrics
                            st.markdown("#### 📱 UX Metrics")
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
                            st.error("Failed to make prediction. Please check your model file.")
    
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
            st.error("Could not load model information. Please check if 'trained_models.pkl' exists.")
    
    elif option == "About":
        st.header("📋 About")
        
        st.markdown("""
        ## Smart Pack Prediction System
        
        This web application uses machine learning models to predict various aspects of smart packaging based on user inputs.
        
        ### Features:
        - **Multiple Input Methods**: Upload CSV files or use manual form input
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
        2. **Gender**: User's gender
        3. **Temperature of Food**: Food temperature preference
        4. **Texture of Food**: Food texture preference
        5. **Stacking**: Packaging stacking preference
        
        ### How to Use:
        1. Choose "Make Predictions" from the sidebar
        2. Either upload a CSV file or use the manual input form
        3. Click "Make Predictions" to get results
        4. Download the results as a CSV file
        
        ### Model Information:
        View detailed information about each trained model including accuracy scores and available prediction classes.
        """)

if __name__ == "__main__":
    main()