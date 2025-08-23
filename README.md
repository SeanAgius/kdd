# Smart Pack Prediction System

A machine learning web application for predicting smart packaging parameters and user experience metrics.

## Features

- **Automation Parameters Prediction**: Type of Sanitiser, No. of Vents, Vents Sizes, Flut Size
- **UX Metrics Prediction**: Packaging Leaking, Package Satisfaction, Sanitiser Satisfaction
- **Multiple Input Methods**: CSV file upload or manual form input
- **Interactive Web Interface**: Built with Streamlit
- **Model Information**: View accuracy and available classes for each model

## Demo

🌐 **Live App**: [Your Streamlit App URL will be here]

## Files Required for Deployment

- `app.py` - Main Streamlit application
- `trained_models.pkl` - Pre-trained machine learning models
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Input Features

The system requires these 5 input features:
1. Age Bracket
2. Gender  
3. Temperature of food
4. Texture of food
5. Stacking

## Output Predictions

### 🤖 Automation Parameters
- Type of Sanitiser
- No. of Vents  
- Vents Sizes
- Flut Size

### 📱 UX Metrics
- Packaging Leaking
- Package Satisfaction
- Sanitiser Satisfaction