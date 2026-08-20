import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import json
from datetime import datetime

st.set_page_config(page_title="FRAMES Security Testing Lab", layout="wide", initial_sidebar_state="expanded")

st.title("🔬 FRAMES Webcam Security Testing Lab")
st.write("Real-time penetration testing with live camera input")

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

# Sidebar settings
st.sidebar.header("⚙️ Testing Configuration")
test_mode = st.sidebar.radio(
    "Select Testing Mode",
    ["Live Camera Testing", "Image Upload Testing", "Model Info"]
)

st.sidebar.divider()
confidence_threshold = st.sidebar.slider("Face Detection Threshold", 0.0, 1.0, 0.5)
spoofing_threshold = st.sidebar.slider("Anti-Spoofing Threshold", 0.0, 1.0, 0.5)
show_raw_output = st.sidebar.checkbox("Show Raw Model Outputs", value=False)

if test_mode == "Live Camera Testing":
    st.subheader("📹 Live Camera Testing")
    st.write("Point your webcam at your face or test object. Watch real-time detection and anti-spoofing analysis.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Camera input
        camera_input = st.camera_input("Capture Image")
    
    with col2:
        st.subheader("⚡ Quick Stats")
        placeholder_face = st.empty()
        placeholder_live = st.empty()
        placeholder_verdict = st.empty()
    
    if camera_input is not None:
        image = Image.open(camera_input)
        img_array = np.array(image)
        
        # Display captured image
        st.image(image, caption="Captured Frame", use_column_width=True)
        
        if st.button("🔍 Analyze Captured Frame"):
            st.subheader("📊 Real-Time Analysis")
            
            # Face Detection
            st.write("### Step 1: Face Detection")
            face_input_details = face_detector.get_input_details()
            face_output_details = face_detector.get_output_details()
            
            input_shape = face_input_details[0]['shape']
            img_resized = cv2.resize(img_array, (input_shape[2], input_shape[1]))
            
            if len(img_resized.shape) == 2:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            
            if face_input_details[0]['dtype'] == np.float32:
                img_input = (img_resized.astype(np.float32) / 255.0)
            else:
                img_input = img_resized.astype(np.uint8)
            
            img_input = np.expand_dims(img_input, axis=0)
            
            face_detector.set_tensor(face_input_details[0]['index'], img_input)
            face_detector.invoke()
            face_output = face_detector.get_tensor(face_output_details[0]['index'])
            
            face_confidence = float(np.max(face_output))
            face_detected = face_confidence > confidence_threshold
            
            # Display face detection results
            col_fd1, col_fd2 = st.columns(2)
            with col_fd1:
                st.metric("Face Confidence", f"{face_confidence:.4f}")
            with col_fd2:
                st.metric("Status", "✅ DETECTED" if face_detected else "❌ NOT DETECTED")
            
            st.write(f"**Threshold:** {confidence_threshold:.2f}")
            st.progress(min(face_confidence, 1.0), text=f"Confidence: {face_confidence:.1%}")
            
            if show_raw_output:
                st.write("**Raw Statistics:**")
                st.write(f"- Min: {np.min(face_output):.6f}")
                st.write(f"- Max: {np.max(face_output):.6f}")
                st.write(f"- Mean: {np.mean(face_output):.6f}")
                st.write(f"- Std: {np.std(face_output):.6f}")
                st.write(f"**Full Output:** {face_output.flatten()}")
            
            st.divider()
            
            # Anti-Spoofing
            st.write("### Step 2: Anti-Spoofing (Liveness Check)")
            
            anti_spoof_input_details = anti_spoofing.get_input_details()
            anti_spoof_output_details = anti_spoofing.get_output_details()
            
            anti_spoof_shape = anti_spoof_input_details[0]['shape']
            img_spoof = cv2.resize(img_array, (anti_spoof_shape[2], anti_spoof_shape[1]))
            
            if len(img_spoof.shape) == 2:
                img_spoof = cv2.cvtColor(img_spoof, cv2.COLOR_GRAY2RGB)
            
            if anti_spoof_input_details[0]['dtype'] == np.float32:
                img_spoof_input = (img_spoof.astype(np.float32) / 255.0)
            else:
                img_spoof_input = img_spoof.astype(np.uint8)
            
            img_spoof_input = np.expand_dims(img_spoof_input, axis=0)
            
            anti_spoofing.set_tensor(anti_spoof_input_details[0]['index'], img_spoof_input)
            anti_spoofing.invoke()
            anti_spoof_output = anti_spoofing.get_tensor(anti_spoof_output_details[0]['index'])
            
            spoofing_score = float(np.max(anti_spoof_output))
            is_live = spoofing_score > spoofing_threshold
            
            # Display anti-spoofing results
            col_as1, col_as2 = st.columns(2)
            with col_as1:
                st.metric("Liveness Score", f"{spoofing_score:.4f}")
            with col_as2:
                st.metric("Status", "✅ REAL FACE" if is_live else "❌ FAKE/PHOTO")
            
            st.write(f"**Threshold:** {spoofing_threshold:.2f}")
            st.progress(min(spoofing_score, 1.0), text=f"Liveness: {spoofing_score:.1%}")
            
            if show_raw_output:
                st.write("**Raw Statistics:**")
                st.write(f"- Min: {np.min(anti_spoof_output):.6f}")
                st.write(f"- Max: {np.max(anti_spoof_output):.6f}")
                st.write(f"- Mean: {np.mean(anti_spoof_output):.6f}")
                st.write(f"- Std: {np.std(anti_spoof_output):.6f}")
                st.write(f"**Full Output:** {anti_spoof_output.flatten()}")
            
            st.divider()
            
            # Final Verdict
            st.subheader("🔐 Final Verdict")
            if face_detected and is_live:
                st.success("✅ ✅ VERIFICATION PASSED")
                st.write("Both face detection and anti-spoofing checks passed!")
            else:
                st.error("❌ ❌ VERIFICATION FAILED")
                if not face_detected:
                    st.write("❌ Face not detected")
                if not is_live:
                    st.write("❌ Face appears fake/photo")
            
            # Export
            st.divider()
            results = {
                "timestamp": datetime.now().isoformat(),
                "face_detection": {
                    "confidence": float(face_confidence),
                    "detected": bool(face_detected),
                    "threshold": confidence_threshold
                },
                "anti_spoofing": {
                    "score": float(spoofing_score),
                    "is_live": bool(is_live),
                    "threshold": spoofing_threshold
                },
                "verdict": "PASS" if (face_detected and is_live) else "FAIL"
            }
            
            st.download_button(
                label="📥 Download Results (JSON)",
                data=json.dumps(results, indent=2),
                file_name=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

elif test_mode == "Image Upload Testing":
    st.subheader("📸 Image Upload Testing")
    st.write("Upload an image file to test")
    
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Analyze Image"):
            with col2:
                st.subheader("Analysis Results")
                
                # Face Detection
                st.write("### Face Detection")
                face_input_details = face_detector.get_input_details()
                face_output_details = face_detector.get_output_details()
                
                input_shape = face_input_details[0]['shape']
                img_resized = cv2.resize(img_array, (input_shape[2], input_shape[1]))
                
                if len(img_resized.shape) == 2:
                    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
                
                if face_input_details[0]['dtype'] == np.float32:
                    img_input = (img_resized.astype(np.float32) / 255.0)
                else:
                    img_input = img_resized.astype(np.uint8)
                
                img_input = np.expand_dims(img_input, axis=0)
                
                face_detector.set_tensor(face_input_details[0]['index'], img_input)
                face_detector.invoke()
                face_output = face_detector.get_tensor(face_output_details[0]['index'])
                
                face_confidence = float(np.max(face_output))
                face_detected = face_confidence > confidence_threshold
                
                st.metric("Face Confidence", f"{face_confidence:.4f}", 
                         "✅ DETECTED" if face_detected else "❌ NOT DETECTED")
                
                if show_raw_output:
                    st.text(f"Raw output: {face_output.flatten()}")
                
                # Anti-Spoofing
                st.write("### Anti-Spoofing")
                anti_spoof_input_details = anti_spoofing.get_input_details()
                anti_spoof_output_details = anti_spoofing.get_output_details()
                
                anti_spoof_shape = anti_spoof_input_details[0]['shape']
                img_spoof = cv2.resize(img_array, (anti_spoof_shape[2], anti_spoof_shape[1]))
                
                if len(img_spoof.shape) == 2:
                    img_spoof = cv2.cvtColor(img_spoof, cv2.COLOR_GRAY2RGB)
                
                if anti_spoof_input_details[0]['dtype'] == np.float32:
                    img_spoof_input = (img_spoof.astype(np.float32) / 255.0)
                else:
                    img_spoof_input = img_spoof.astype(np.uint8)
                
                img_spoof_input = np.expand_dims(img_spoof_input, axis=0)
                
                anti_spoofing.set_tensor(anti_spoof_input_details[0]['index'], img_spoof_input)
                anti_spoofing.invoke()
                anti_spoof_output = anti_spoofing.get_tensor(anti_spoof_output_details[0]['index'])
                
                spoofing_score = float(np.max(anti_spoof_output))
                is_live = spoofing_score > spoofing_threshold
                
                st.metric("Liveness Score", f"{spoofing_score:.4f}", 
                         "✅ REAL" if is_live else "❌ FAKE")
                
                if show_raw_output:
                    st.text(f"Raw output: {anti_spoof_output.flatten()}")
                
                # Verdict
                st.divider()
                if face_detected and is_live:
                    st.success("✅ VERIFICATION PASSED")
                else:
                    st.error("❌ VERIFICATION FAILED")

elif test_mode == "Model Info":
    st.subheader("📊 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Face Detection Model**")
        face_inputs = face_detector.get_input_details()
        face_outputs = face_detector.get_output_details()
        
        st.json({
            "Input Shape": str(face_inputs[0]['shape']),
            "Input Type": str(face_inputs[0]['dtype']),
            "Output Shape": str(face_outputs[0]['shape']),
            "Output Type": str(face_outputs[0]['dtype']),
        })
    
    with col2:
        st.write("**Anti-Spoofing Model**")
        spoof_inputs = anti_spoofing.get_input_details()
        spoof_outputs = anti_spoofing.get_output_details()
        
        st.json({
            "Input Shape": str(spoof_inputs[0]['shape']),
            "Input Type": str(spoof_inputs[0]['dtype']),
            "Output Shape": str(spoof_outputs[0]['shape']),
            "Output Type": str(spoof_outputs[0]['dtype']),
        })

with st.sidebar:
    st.divider()
    st.subheader("📖 How to Use")
    st.markdown("""
    ### Live Camera Testing:
    1. Click "Capture Image" to take a photo
    2. Click "Analyze" to see results
    3. Try different things:
       - Your real face
       - A printed photo
       - Your face with mask
       - Different lighting
       - Blurry images
    
    ### Upload Testing:
    1. Upload an image file
    2. Click analyze
    3. See detection & anti-spoofing scores
    
    ### What to Test:
    - ✅ Real faces
    - 📷 Printed photos
    - 😷 Masks/covering
    - 🌞 Different lighting
    - 📸 Blurry/pixelated images
    - 🪞 Mirrors/reflections
    - 🎥 Video on screen
    """)
