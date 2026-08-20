import streamlit as st
import numpy as np
import cv2
from PIL import Image
import json
from datetime import datetime
import subprocess
import sys

# Try to install tensorflow-lite-runtime if not available
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    except ImportError:
        # Fallback: create a simple stub
        st.error("❌ TensorFlow Lite runtime not available. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tflite-runtime"])
        from tflite_runtime.interpreter import Interpreter

st.set_page_config(page_title="FRAMES Verification System", layout="wide", initial_sidebar_state="expanded")

st.title("🎓 FRAMES Teacher Attendance Verification System")
st.write("Local testing interface for FRAMES face detection and anti-spoofing algorithm")

# Load models
@st.cache_resource
def load_models():
    try:
        face_detector = Interpreter(model_path="models/face_model.tflite")
        face_detector.allocate_tensors()
        
        anti_spoofing = Interpreter(model_path="models/FaceAntiSpoofing.tflite")
        anti_spoofing.allocate_tensors()
        
        return face_detector, anti_spoofing
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

try:
    face_detector, anti_spoofing = load_models()
    if face_detector and anti_spoofing:
        st.success("✅ Models loaded successfully!")
    else:
        st.stop()
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# Sidebar for settings
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
spoofing_threshold = st.sidebar.slider("Anti-Spoofing Threshold", 0.0, 1.0, 0.5)

# Create two columns for upload and live camera
col1, col2 = st.columns(2)

with col1:
    st.subheader("📸 Upload Face Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="upload")

# Store results
results = None

def process_face_image(image_array):
    """Complete FRAMES verification workflow"""
    
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "status": "processing",
        "face_detected": False,
        "face_quality": None,
        "is_live": False,
        "spoofing_score": None,
        "verification_result": "FAILED",
        "errors": []
    }
    
    try:
        # Step 1: Face Detection
        st.info("Step 1️⃣: Detecting face...")
        face_input_details = face_detector.get_input_details()
        face_output_details = face_detector.get_output_details()
        
        input_shape = face_input_details[0]['shape']
        img_resized = cv2.resize(image_array, (input_shape[2], input_shape[1]))
        
        # Handle grayscale/color
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        
        # Normalize
        if face_input_details[0]['dtype'] == np.float32:
            img_input = (img_resized.astype(np.float32) / 255.0)
        else:
            img_input = img_resized.astype(np.uint8)
        
        img_input = np.expand_dims(img_input, axis=0)
        
        # Run face detection
        face_detector.set_tensor(face_input_details[0]['index'], img_input)
        face_detector.invoke()
        face_output = face_detector.get_tensor(face_output_details[0]['index'])
        
        # Parse face detection output
        face_confidence = float(np.max(face_output))
        results_dict["face_detected"] = face_confidence > confidence_threshold
        results_dict["face_quality"] = float(face_confidence)
        
        if not results_dict["face_detected"]:
            results_dict["errors"].append(f"No face detected (confidence: {face_confidence:.2%})")
            st.warning(f"⚠️ No face detected with sufficient confidence (detected: {face_confidence:.2%})")
            return results_dict
        
        st.success(f"✅ Face detected with {face_confidence:.2%} confidence")
        
        # Step 2: Anti-Spoofing Check
        st.info("Step 2️⃣: Checking if face is real (anti-spoofing)...")
        anti_spoof_input_details = anti_spoofing.get_input_details()
        anti_spoof_output_details = anti_spoofing.get_output_details()
        
        anti_spoof_input_shape = anti_spoof_input_details[0]['shape']
        img_resized_spoof = cv2.resize(image_array, (anti_spoof_input_shape[2], anti_spoof_input_shape[1]))
        
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
        
        # Parse anti-spoofing output (higher = more likely real face)
        spoofing_score = float(np.max(anti_spoof_output))
        results_dict["spoofing_score"] = spoofing_score
        results_dict["is_live"] = spoofing_score > spoofing_threshold
        
        if not results_dict["is_live"]:
            results_dict["errors"].append(f"Spoofing detected (liveness score: {spoofing_score:.2%})")
            st.error(f"🚫 Spoofing detected! Face appears to be fake/photo (score: {spoofing_score:.2%})")
            return results_dict
        
        st.success(f"✅ Real face detected (liveness score: {spoofing_score:.2%})")
        
        # Step 3: Final Verification
        if results_dict["face_detected"] and results_dict["is_live"]:
            results_dict["status"] = "verified"
            results_dict["verification_result"] = "PASSED"
            st.success("✅ ✅ VERIFICATION PASSED - Teacher attendance recorded!")
        
    except Exception as e:
        results_dict["errors"].append(str(e))
        st.error(f"❌ Error during verification: {e}")
    
    return results_dict

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    with col1:
        st.image(image, use_column_width=True, caption="Uploaded Image")
    
    # Process image
    if st.button("🔍 Run Verification", key="verify"):
        with col2:
            st.subheader("📋 Verification Results")
            results = process_face_image(img_array)
            
            # Display results
            st.divider()
            col_status1, col_status2 = st.columns(2)
            
            with col_status1:
                st.metric("Verification Status", results["verification_result"], 
                         "✅ PASSED" if results["verification_result"] == "PASSED" else "❌ FAILED")
            
            with col_status2:
                st.metric("Timestamp", results["timestamp"].split('T')[1][:8])
            
            # Detailed results
            st.write("**Detailed Analysis:**")
            df_results = {
                "Face Detected": "✅ Yes" if results["face_detected"] else "❌ No",
                "Face Confidence": f"{results['face_quality']*100:.1f}%" if results['face_quality'] else "N/A",
                "Is Live (Anti-Spoofing)": "✅ Yes" if results["is_live"] else "❌ No",
                "Liveness Score": f"{results['spoofing_score']*100:.1f}%" if results['spoofing_score'] else "N/A",
            }
            
            for key, value in df_results.items():
                st.write(f"• {key}: {value}")
            
            if results["errors"]:
                st.error("**Errors:**")
                for error in results["errors"]:
                    st.write(f"- {error}")
            
            # JSON export
            st.divider()
            st.subheader("📊 Raw Output (JSON)")
            st.json(results)

st.divider()
st.write("---")

# Info sidebar
with st.sidebar:
    st.subheader("ℹ️ How It Works")
    st.markdown("""
    ### FRAMES Verification Flow:
    
    1. **Face Detection** 🎭
       - Detects if a face is present in image
       - Outputs confidence score
    
    2. **Anti-Spoofing** 🛡️
       - Checks if face is real (not a photo/video)
       - Prevents fraud with fake images
    
    3. **Verification** ✅
       - Both checks must pass for attendance
       - Records timestamp and results
    
    ### Model Details:
    - **face_model.tflite**: Face detection
    - **FaceAntiSpoofing.tflite**: Liveness detection
    
    ### Research Notes:
    This is a local research interface for testing the FRAMES teacher attendance algorithm.
    """)
