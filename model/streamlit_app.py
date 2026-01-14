
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import json

class HybridViTModel(nn.Module):
    def __init__(self, num_classes=6, transformer_depth=4, dropout=0.1):
        super(HybridViTModel, self).__init__()
        from torchvision import models
        self.resnet = models.resnet50(weights=None)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        
        self.feature_dim = 2048
        self.patch_size = 7
        self.num_patches = (224 // self.patch_size) ** 2
        self.patch_embed_dim = 512
        
        self.patch_embed = nn.Conv2d(
            self.feature_dim, self.patch_embed_dim, 
            kernel_size=self.patch_size, stride=self.patch_size
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.patch_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.patch_embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.patch_embed_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_depth)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.patch_embed_dim, num_classes)
        
    def forward(self, x):
        batch_size = x.size(0)
        features = self.resnet(x)
        patch_embeds = self.patch_embed(features)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, patch_embeds), dim=1)
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed[:, :x.size(1), :]
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed
        x = self.dropout(x)
        x = self.transformer(x)
        cls_output = x[:, 0]
        output = self.classifier(cls_output)
        return output

@st.cache_resource
def load_model():
    model = HybridViTModel(num_classes=6)
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image):
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    image = np.array(image.convert('RGB'))
    transformed = transform(image=image)
    return transformed['image'].unsqueeze(0)

st.set_page_config(page_title="Skin Disease Classifier", layout="wide")
st.title("🧬 Skin Disease Classification App")
st.write("Upload an image of a skin lesion to classify it into one of 6 categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        with st.spinner('Classifying...'):
            model = load_model()
            processed_image = preprocess_image(image)
            
            with torch.no_grad():
                outputs = model(processed_image)
                probabilities = torch.softmax(outputs, dim=1)
            
            probs = probabilities[0].numpy()
            class_names = ["Chickenpox", "Cowpox", "HFMD", "Healthy", "Measles", "Monkeypox"]
            
            top3_indices = np.argsort(probs)[-3:][::-1]
            top3_classes = [class_names[i] for i in top3_indices]
            top3_probs = [probs[i] for i in top3_indices]
            
            st.subheader("🔍 Prediction Results")
            st.success(f"**Predicted Class:** {top3_classes[0]} ({top3_probs[0]*100:.2f}%)")
            
            st.subheader("📊 Top 3 Predictions")
            for i, (cls, prob) in enumerate(zip(top3_classes, top3_probs)):
                st.write(f"{i+1}. {cls}: {prob*100:.2f}%")
            
            st.subheader("📈 All Class Probabilities")
            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(len(class_names))
            ax.barh(y_pos, probs * 100, color='skyblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(class_names)
            ax.set_xlabel('Probability (%)')
            ax.set_title('Class Probabilities')
            ax.invert_yaxis()
            st.pyplot(fig)

st.sidebar.header("ℹ️ Instructions")
st.sidebar.write("""
1. Upload a clear image of a skin lesion
2. Wait for the model to process the image
3. View the prediction results and probabilities
4. The app shows the top 3 most likely diagnoses

**Note:** This is for educational purposes only. 
Always consult a healthcare professional for medical diagnosis.
""")

st.sidebar.header("🏥 Disease Classes")
st.sidebar.write("""
- Chickenpox
- Cowpox  
- HFMD (Hand Foot Mouth Disease)
- Healthy
- Measles
- Monkeypox
""")
