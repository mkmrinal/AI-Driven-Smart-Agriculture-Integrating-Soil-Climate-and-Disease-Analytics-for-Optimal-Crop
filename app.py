import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Crop Recommendation System ---
def load_crop_recommendation_model():
    model = joblib.load(r"C:\Users\mmukh\OneDrive\Desktop\Major project\Coding\Recommendation\recommendation_model.pkl")
    df = pd.read_csv(r"C:\Users\mmukh\OneDrive\Desktop\Major project\Data Sets\Crop_recommendation.csv")
    
    try:
        label_encoder = joblib.load(r"C:\Users\mmukh\OneDrive\Desktop\Major project\Coding\Recommendation\label_encoder.pkl")
    except FileNotFoundError:
        label_encoder = LabelEncoder()
        df["label"] = label_encoder.fit_transform(df["label"])
        joblib.dump(label_encoder, r"C:\Users\mmukh\OneDrive\Desktop\Major project\Coding\Recommendation\label_encoder.pkl")
    
    return model, label_encoder

def get_user_input():
    st.sidebar.header("Soil & Weather Conditions")
    
    N = st.sidebar.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
    P = st.sidebar.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
    K = st.sidebar.number_input("Potassium (K)", min_value=0, max_value=200, value=50)
    temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, value=25.0)
    humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    ph = st.sidebar.number_input("Soil pH", min_value=0.0, max_value=14.0, value=7.0)
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0)
    
    return np.array([[N, P, K, temperature, humidity, ph, rainfall]])

# --- Disease Prediction System ---
def load_disease_model():
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])
    
    data_path = r"C:\Users\mmukh\OneDrive\Desktop\Major project\Data Sets\plantvillage dataset\color"
    class_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names)
    
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.01),
        nn.Linear(model.fc.in_features, len(label_encoder.classes_))
    )
    model = model.to(device)
    
    model_path = r"C:\Users\mmukh\OneDrive\Desktop\Major project\Coding\Prediction\plant_disease_resnet50.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, label_encoder, transform

def predict_disease(image, model, label_encoder, transform):
    try:
        img = Image.open(image).convert('RGB')
        img_transformed = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_transformed)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
            confidence = confidence.item()
            
            top3_probs, top3_indices = torch.topk(probabilities, 3)
            top3_classes = label_encoder.inverse_transform(top3_indices.cpu().numpy()[0])
            top3_confidences = top3_probs.cpu().numpy()[0]
            top_predictions = list(zip(top3_classes, top3_confidences))
        
        return predicted_class, confidence, top_predictions, img
    
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None, None, None

def main():
    st.title("AgriSmart: Crop & Disease Analysis System")
    
    crop_model, crop_label_encoder = load_crop_recommendation_model()
    disease_model, disease_label_encoder, disease_transform = load_disease_model()
    
    tab1, tab2 = st.tabs(["Crop Recommendation", "Disease Detection"])
    
    with tab1:
        st.header("Crop Recommendation")
        st.write("Enter soil and weather conditions to get the best crop recommendation.")
        
        data = get_user_input()
        
        if st.button("Get Crop Recommendation"):
            prediction = crop_model.predict(data)
            predicted_crop = crop_label_encoder.inverse_transform(prediction)[0]
            st.success(f"**Recommended Crop:** {predicted_crop}")
    
    with tab2:
        st.header("Plant Disease Detection")
        st.write("Upload an image of a plant leaf to detect potential diseases.")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', width=300)
            
            if st.button("Analyze Disease"):
                with st.spinner('Analyzing...'):
                    predicted_class, confidence, top_predictions, _ = predict_disease(
                        uploaded_file, disease_model, disease_label_encoder, disease_transform
                    )
                
                if predicted_class:
                    st.success(f"**Prediction:** {predicted_class.replace('_', ' ')}")
                    st.success(f"**Confidence:** {confidence:.2%}")
                    
                    st.subheader("Top Predictions:")
                    for i, (class_name, prob) in enumerate(top_predictions, 1):
                        st.write(f"{i}. {class_name.replace('_', ' '):<30} {prob:.2%}")
                    
                    if "healthy" in predicted_class:
                        st.balloons()
                        st.success("This plant appears to be healthy!")
                    else:
                        st.warning("This plant may have a disease. Please consult with an agricultural expert.")

if __name__ == "__main__":
    main()