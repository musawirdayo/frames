import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

st.set_page_config(page_title="Face Detection & Anti-Spoofing", layout="wide")

st.title("🎭 Face Detection & Anti-Spoofing App")
st.write("Upload an image to detect faces and check for spoofing attacks")

# Load models
@st.cache_resource
def load_models():
    face_detector = tf.lite.Interpreter(model_path="models/face_model.tflite")
    face_detector.allocate_tensors()
    
    anti_spoofing = tf.lite.Interpreter(model_path="models/FaceAntiSpoofing.tflite")
    anti_spoofing.allocate_tensors()
    
    return face_detector, anti_spoofing

try:
    face_detector, anti_spoofing = load_models()
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    # Convert PIL image to numpy array for processing
    img_array = np.array(image)
    
    # Get model details
    face_input_details = face_detector.get_input_details()
    face_output_details = face_detector.get_output_details()
    
    anti_spoof_input_details = anti_spoofing.get_input_details()
    anti_spoof_output_details = anti_spoofing.get_output_details()
    
    with col2:
        st.subheader("Model Information")
        st.write("**Face Detection Model:**")
        st.write(f"  - Input shape: {face_input_details[0]['shape']}")
        st.write(f"  - Output shape: {face_output_details[0]['shape']}")
        
        st.write("**Anti-Spoofing Model:**")
        st.write(f"  - Input shape: {anti_spoof_input_details[0]['shape']}")
        st.write(f"  - Output shape: {anti_spoof_output_details[0]['shape']}")
    
    st.divider()
    
    # Process image
    if st.button("🔍 Analyze Image", key="analyze"):
        st.subheader("Analysis Results")
        
        try:
            # Prepare input for face detection
            input_shape = face_input_details[0]['shape']
            img_resized = cv2.resize(img_array, (input_shape[2], input_shape[1]))
            
            if len(img_resized.shape) == 2:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            
            # Normalize if needed
            if face_input_details[0]['dtype'] == np.float32:
                img_input = (img_resized.astype(np.float32) / 255.0)
            else:
                img_input = img_resized.astype(np.uint8)
            
            # Add batch dimension
            img_input = np.expand_dims(img_input, axis=0)
            
            # Run face detection
            face_detector.set_tensor(face_input_details[0]['index'], img_input)
            face_detector.invoke()
            face_output = face_detector.get_tensor(face_output_details[0]['index'])
            
            st.write(f"**Face Detection Output Shape:** {face_output.shape}")
            st.write(f"**Face Detection Output Sample:** {face_output.flatten()[:5]}...")
            
            # Run anti-spoofing detection if applicable
            anti_spoof_input_shape = anti_spoof_input_details[0]['shape']
            img_resized_spoof = cv2.resize(img_array, (anti_spoof_input_shape[2], anti_spoof_input_shape[1]))
            
            if len(img_resized_spoof.shape) == 2:
                img_resized_spoof = cv2.cvtColor(img_resized_spoof, cv2.COLOR_GRAY2RGB)
            
            if anti_spoof_input_details[0]['dtype'] == np.float32:
                img_spoof_input = (img_resized_spoof.astype(np.float32) / 255.0)
            else:
                img_spoof_input = img_resized_spoof.astype(np.uint8)
            
            img_spoof_input = np.expand_dims(img_spoof_input, axis=0)
            
            anti_spoofing.set_tensor(anti_spoof_input_details[0]['index'], img_spoof_input)
            anti_spoofing.invoke()
            anti_spoof_output = anti_spoofing.get_tensor(anti_spoof_output_details[0]['index'])
            
            st.write(f"**Anti-Spoofing Output Shape:** {anti_spoof_output.shape}")
            st.write(f"**Anti-Spoofing Output Sample:** {anti_spoof_output.flatten()[:5]}...")
            
            st.success("✅ Analysis completed!")
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
            st.write("Please check the image format and model inputs.")

st.divider()
st.write("---")
st.write("**App Features:**")
st.write("- 📸 Face detection using TensorFlow Lite")
st.write("- 🎭 Anti-spoofing detection")
st.write("- 🖼️ Real-time image processing")
