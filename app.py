import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, BatchNormalization
import cv2
import os

# Page configuration
st.set_page_config(
    page_title="Real vs Fake Image Detection",
    page_icon="🔍",
    layout="centered"
)

st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .real-result {
        background: linear-gradient(135deg, #a8e6cf 0%, #c8f7dc 100%);
        border: 3px solid #28a745;
    }
    .real-result h2 {
        color: #155724;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    .fake-result {
        background: linear-gradient(135deg, #ffb3ba 0%, #ffd6d9 100%);
        border: 3px solid #dc3545;
    }
    .fake-result h2 {
        color: #721c24;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    .result-box p {
        color: #000;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🔍 Real vs Fake Image Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload an image to detect if it\'s real or deepfake/AI-generated</p>', unsafe_allow_html=True)

# Sidebar information
with st.sidebar:
    st.header("About")
    st.write("This application uses Xception deep learning model to detect whether an image is real or deepfake/AI-generated.")
    st.write("---")
    st.header("How to use")
    st.write("1. Upload an image using the file uploader")
    st.write("2. Click 'Analyze Image' button")
    st.write("3. View the prediction results")

# Different architecture configurations to try
def build_arch_v1(input_shape=(299, 299, 3)):
    """Standard Xception + GAP + Dense"""
    base = Xception(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base.input, outputs=x)

def build_arch_v2(input_shape=(299, 299, 3)):
    """Xception + GAP + Dense(128) + Dense(1)"""
    base = Xception(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base.input, outputs=x)

def build_arch_v3(input_shape=(299, 299, 3)):
    """Xception + GAP + Dense(256) + Dense(1)"""
    base = Xception(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base.input, outputs=x)

def build_arch_v4(input_shape=(299, 299, 3)):
    """Xception + GAP + Dense(512) + Dense(1)"""
    base = Xception(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base.input, outputs=x)

def build_arch_v5(input_shape=(299, 299, 3)):
    """Xception + GAP + Dense(1024) + Dense(1)"""
    base = Xception(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base.input, outputs=x)

def build_arch_v6(input_shape=(299, 299, 3)):
    """Xception + GAP + Multiple Dense layers"""
    base = Xception(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base.input, outputs=x)

def build_arch_v7(input_shape=(299, 299, 3)):
    """Xception + Flatten + Dense"""
    base = Xception(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base.input, outputs=x)

def build_arch_v8(input_shape=(299, 299, 3)):
    """Xception + GAP + BatchNorm + Dense"""
    base = Xception(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base.input, outputs=x)

def build_arch_v9(input_shape=(299, 299, 3)):
    """Xception + GAP + Dense(2) softmax"""
    base = Xception(include_top=False, weights='imagenet', input_shape=input_shape)
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2, activation='softmax')(x)
    return Model(inputs=base.input, outputs=x)

def build_arch_v10(input_shape=(299, 299, 3)):
    """Xception trainable=False + top layers"""
    base = Xception(include_top=False, weights='imagenet', input_shape=input_shape)
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(inputs=base.input, outputs=x)

# Load model with auto-detection
@st.cache_resource
def load_model():
    """Try multiple architectures automatically"""
    
    # Find weights file
    possible_paths = [
        'E:/miniproject/New folder/xceptionnew.weights.h5',
        'E:/miniproject/New folder/xceptionnew-weights.h5',
        'E:/miniproject/New folder/model.h5',
        'xceptionnew.weights.h5',
        'model.h5',
    ]
    
    weights_path = None
    for path in possible_paths:
        if os.path.exists(path):
            weights_path = path
            st.sidebar.info(f"📁 Found: {os.path.basename(path)}")
            break
    
    if not weights_path:
        st.sidebar.error("❌ Weights file not found!")
        return None
    
    # List of architectures to try
    architectures = [
        #("Simple: GAP + Dense(1)", build_arch_v1),
        #("Standard: GAP + Dense(128)", build_arch_v2),
        #("Medium: GAP + Dense(256)", build_arch_v3),
        ("Large: GAP + Dense(512)", build_arch_v4),
        ("XLarge: GAP + Dense(1024)", build_arch_v5),
        ("Multi-layer: 512→256→1", build_arch_v6),
        ("Flatten approach", build_arch_v7),
        ("With BatchNorm", build_arch_v8),
        ("Softmax output (2 classes)", build_arch_v9),
        ("Frozen base model", build_arch_v10),
    ]
    
    # Try each architecture
    for idx, (name, build_func) in enumerate(architectures, 1):
        try:
            st.sidebar.info(f"🔄 Trying {idx}/10: {name}")
            
            model = build_func(input_shape=(299, 299, 3))
            
            # Try different loading methods
            try:
                # Method 1: Load with skip_mismatch
                model.load_weights(weights_path, skip_mismatch=True)
            except:
                try:
                    # Method 2: Load normally
                    model.load_weights(weights_path)
                except:
                    # Method 3: Load by name
                    model.load_weights(weights_path, by_name=True)
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            st.sidebar.success(f"✅ SUCCESS with: {name}")
            return model
            
        except Exception as e:
            error_msg = str(e)[:80]
            st.sidebar.warning(f"❌ Failed: {error_msg}")
            continue
    
    # All failed
    st.sidebar.error("❌ Could not load with any architecture!")
    st.sidebar.info("""
    **Need your help!**
    
    Please share your Colab training code, specifically:
    - How you added layers after Xception
    - What Dense layer sizes you used
    - Number of output neurons (1 or 2)
    
    Or upload the full model file (not just weights)
    """)
    return None

# Preprocess image
def preprocess_image(image, target_size=(299, 299)):
    """Preprocess for Xception"""
    try:
        img_array = np.array(image)
        
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        img_resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_AREA)
        img_float = img_resized.astype('float32')
        
        # Xception preprocessing
        img_preprocessed = (img_float / 127.5) - 1.0
        img_batch = np.expand_dims(img_preprocessed, axis=0)
        
        return img_batch
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None

# Predict
def predict_image(model, image):
    """Make prediction"""
    try:
        img_array = preprocess_image(image)
        if img_array is None:
            return "ERROR", 0.0
        
        predictions = model.predict(img_array, verbose=0)
        
        # Handle different output formats
        if predictions.shape[1] > 1:
            # Softmax output [fake, real]
            real_prob = float(predictions[0][1])
            fake_prob = float(predictions[0][0])
            
            if real_prob > fake_prob:
                return "REAL", real_prob
            else:
                return "DEEPFAKE", fake_prob
        else:
            # Sigmoid output
            score = float(predictions[0][0])
            
            if score > 0.5:
                return "REAL", score
            else:
                return "DEEPFAKE", 1 - score
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "ERROR", 0.0

# Main app
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['jpg', 'jpeg', 'png', 'bmp'],
    help="Upload an image"
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, width=400)
        
        with col2:
            st.subheader("Analysis Results")
            
            model = load_model()
            
            if model is None:
                st.error("⚠️ Model loading failed - check sidebar")
            else:
                if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("🤖 Analyzing..."):
                        prediction, confidence = predict_image(model, image)
                        
                        if prediction == "REAL":
                            st.markdown(f"""
                                <div class="result-box real-result">
                                    <h2>✅ REAL IMAGE</h2>
                                    <p style="font-size: 1.5rem; margin: 10px 0;">
                                        Confidence: {confidence*100:.2f}%
                                    </p>
                                    <p>This image appears to be authentic.</p>
                                </div>
                            """, unsafe_allow_html=True)
                        elif prediction == "DEEPFAKE":
                            st.markdown(f"""
                                <div class="result-box fake-result">
                                    <h2>⚠️ DEEPFAKE DETECTED</h2>
                                    <p style="font-size: 1.5rem; margin: 10px 0;">
                                        Confidence: {confidence*100:.2f}%
                                    </p>
                                    <p>This image may be AI-generated or manipulated.</p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error("❌ Prediction failed")
                        
                        if prediction != "ERROR":
                            st.write("---")
                            st.write(f"**Format:** {image.format or 'Unknown'}")
                            st.write(f"**Size:** {image.size[0]}×{image.size[1]} px")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("👆 Upload an image to detect deepfakes")
    st.write("---")
    st.write("**💡 Tips:**")
    st.write("✓ Use clear, high-quality images")
    st.write("✓ Best results with face images")
    st.write("✓ Supports: JPG, PNG, BMP")

st.write("---")
st.markdown('<p style="text-align: center; color: #666;">🔒 Images processed locally</p>', unsafe_allow_html=True)

# Sidebar status
st.sidebar.write("---")
st.sidebar.header("⚙️ Model Status")
model = load_model()
if model:
    st.sidebar.success("✅ Model Ready")
    try:
        st.sidebar.write(f"**Layers:** {len(model.layers)}")
        st.sidebar.write(f"**Parameters:** {model.count_params():,}")
    except:
        pass
else:
    st.sidebar.error("❌ Model Not Loaded")