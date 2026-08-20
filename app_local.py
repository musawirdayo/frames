import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from datetime import datetime
import cv2
import os

st.set_page_config(page_title="FRAMES Security Testing Lab", layout="wide", initial_sidebar_state="expanded")

st.title("🔬 FRAMES Algorithm Security Testing Lab")
st.write("Advanced penetration testing interface for FRAMES face detection and anti-spoofing system")

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

# Sidebar for advanced settings
st.sidebar.header("🛠️ Testing Configuration")
test_mode = st.sidebar.radio(
    "Select Testing Mode",
    ["Single Image Analysis", "Batch Testing", "Model Introspection", "Attack Vector Testing"]
)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Advanced Settings")
raw_output_mode = st.sidebar.checkbox("Show Raw Model Outputs", value=False)
confidence_threshold = st.sidebar.slider("Face Detection Threshold", 0.0, 1.0, 0.5)
spoofing_threshold = st.sidebar.slider("Anti-Spoofing Threshold", 0.0, 1.0, 0.5)

# Get model details
def get_model_specs():
    """Extract detailed model specifications"""
    specs = {}
    
    # Face detector specs
    face_inputs = face_detector.get_input_details()
    face_outputs = face_detector.get_output_details()
    specs['face_detector'] = {
        'inputs': face_inputs,
        'outputs': face_outputs,
        'input_shape': face_inputs[0]['shape'],
        'output_shape': face_outputs[0]['shape'],
        'input_dtype': str(face_inputs[0]['dtype']),
        'output_dtype': str(face_outputs[0]['dtype']),
    }
    
    # Anti-spoofing specs
    spoof_inputs = anti_spoofing.get_input_details()
    spoof_outputs = anti_spoofing.get_output_details()
    specs['anti_spoofing'] = {
        'inputs': spoof_inputs,
        'outputs': spoof_outputs,
        'input_shape': spoof_inputs[0]['shape'],
        'output_shape': spoof_outputs[0]['shape'],
        'input_dtype': str(spoof_inputs[0]['dtype']),
        'output_dtype': str(spoof_outputs[0]['dtype']),
    }
    
    return specs

if test_mode == "Model Introspection":
    st.subheader("📊 Model Architecture Analysis")
    
    specs = get_model_specs()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Face Detection Model**")
        st.json({
            "Input Shape": str(specs['face_detector']['input_shape']),
            "Input Type": specs['face_detector']['input_dtype'],
            "Output Shape": str(specs['face_detector']['output_shape']),
            "Output Type": specs['face_detector']['output_dtype'],
        })
    
    with col2:
        st.write("**Anti-Spoofing Model**")
        st.json({
            "Input Shape": str(specs['anti_spoofing']['input_shape']),
            "Input Type": specs['anti_spoofing']['input_dtype'],
            "Output Shape": str(specs['anti_spoofing']['output_shape']),
            "Output Type": specs['anti_spoofing']['output_dtype'],
        })
    
    st.divider()
    st.write("**Full Model Details**")
    st.json(specs, expanded=False)

elif test_mode == "Single Image Analysis":
    st.subheader("🔍 Single Image Deep Analysis")
    
    uploaded_file = st.file_uploader("Upload test image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, use_column_width=True)
            st.write(f"**Image Info:**")
            st.write(f"- Size: {image.size}")
            st.write(f"- Mode: {image.mode}")
            st.write(f"- Array shape: {img_array.shape}")
        
        if st.button("🔬 Run Full Analysis"):
            with col2:
                st.subheader("📈 Analysis Results")
                
                # Face Detection Analysis
                st.write("### Step 1: Face Detection")
                face_input_details = face_detector.get_input_details()
                face_output_details = face_detector.get_output_details()
                
                input_shape = face_input_details[0]['shape']
                img_resized = cv2.resize(img_array, (input_shape[2], input_shape[1]))
                
                if len(img_resized.shape) == 2:
                    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
                
                # Try different normalizations
                if face_input_details[0]['dtype'] == np.float32:
                    img_input = (img_resized.astype(np.float32) / 255.0)
                else:
                    img_input = img_resized.astype(np.uint8)
                
                img_input = np.expand_dims(img_input, axis=0)
                
                face_detector.set_tensor(face_input_details[0]['index'], img_input)
                face_detector.invoke()
                face_output = face_detector.get_tensor(face_output_details[0]['index'])
                
                st.write(f"**Face Detection Output Shape:** {face_output.shape}")
                st.write(f"**Output Stats:**")
                st.write(f"- Min: {np.min(face_output):.6f}")
                st.write(f"- Max: {np.max(face_output):.6f}")
                st.write(f"- Mean: {np.mean(face_output):.6f}")
                st.write(f"- Std: {np.std(face_output):.6f}")
                
                if raw_output_mode:
                    st.write("**Raw Output Array:**")
                    st.write(face_output)
                    st.write(f"**Flattened Values:** {face_output.flatten()}")
                
                face_confidence = float(np.max(face_output))
                face_detected = face_confidence > confidence_threshold
                
                st.metric("Face Detection Confidence", f"{face_confidence:.4f}", 
                         "✅ DETECTED" if face_detected else "❌ NOT DETECTED")
                
                # Anti-Spoofing Analysis
                st.write("### Step 2: Anti-Spoofing (Liveness) Check")
                
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
                
                st.write(f"**Anti-Spoofing Output Shape:** {anti_spoof_output.shape}")
                st.write(f"**Output Stats:**")
                st.write(f"- Min: {np.min(anti_spoof_output):.6f}")
                st.write(f"- Max: {np.max(anti_spoof_output):.6f}")
                st.write(f"- Mean: {np.mean(anti_spoof_output):.6f}")
                st.write(f"- Std: {np.std(anti_spoof_output):.6f}")
                
                if raw_output_mode:
                    st.write("**Raw Output Array:**")
                    st.write(anti_spoof_output)
                    st.write(f"**Flattened Values:** {anti_spoof_output.flatten()}")
                
                spoofing_score = float(np.max(anti_spoof_output))
                is_live = spoofing_score > spoofing_threshold
                
                st.metric("Liveness Score", f"{spoofing_score:.4f}", 
                         "✅ REAL FACE" if is_live else "❌ FAKE/PHOTO")
                
                # Summary
                st.divider()
                st.subheader("🔐 Security Verdict")
                if face_detected and is_live:
                    st.success("✅ VERIFICATION PASSED")
                else:
                    st.error("❌ VERIFICATION FAILED")
                
                # Exportable results
                st.subheader("📊 JSON Export")
                results = {
                    "timestamp": datetime.now().isoformat(),
                    "face_detection": {
                        "confidence": float(face_confidence),
                        "detected": bool(face_detected),
                        "threshold": confidence_threshold,
                        "raw_stats": {
                            "min": float(np.min(face_output)),
                            "max": float(np.max(face_output)),
                            "mean": float(np.mean(face_output)),
                            "std": float(np.std(face_output))
                        }
                    },
                    "anti_spoofing": {
                        "score": float(spoofing_score),
                        "is_live": bool(is_live),
                        "threshold": spoofing_threshold,
                        "raw_stats": {
                            "min": float(np.min(anti_spoof_output)),
                            "max": float(np.max(anti_spoof_output)),
                            "mean": float(np.mean(anti_spoof_output)),
                            "std": float(np.std(anti_spoof_output))
                        }
                    },
                    "final_verdict": "PASS" if (face_detected and is_live) else "FAIL"
                }
                
                st.json(results)
                
                # Download button
                st.download_button(
                    label="Download Results (JSON)",
                    data=json.dumps(results, indent=2),
                    file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

elif test_mode == "Attack Vector Testing":
    st.subheader("🎯 Attack Vector Testing")
    st.write("Test common bypass techniques used to fool face detection and anti-spoofing systems")
    
    attack_vector = st.selectbox(
        "Select Attack Vector",
        [
            "Brightness/Contrast Manipulation",
            "Blurring/Noise Addition",
            "Face Mask/Covering",
            "Mirror/Reflection",
            "Print Attack (Photo Display)",
            "Video Replay Attack",
            "Custom Attack"
        ]
    )
    
    uploaded_file = st.file_uploader("Upload base image for attack testing", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        if attack_vector == "Brightness/Contrast Manipulation":
            st.write("**Testing brightness/contrast variations to bypass detection**")
            brightness_factor = st.slider("Brightness Factor", 0.3, 2.0, 1.0, 0.1)
            contrast_factor = st.slider("Contrast Factor", 0.3, 2.0, 1.0, 0.1)
            
            # Apply transformations
            img_modified = cv2.convertScaleAbs(img_array, alpha=contrast_factor, beta=brightness_factor)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original")
            with col2:
                st.image(img_modified, caption="Modified")
            
            if st.button("Test Modified Image"):
                # Run inference on modified image
                face_input_details = face_detector.get_input_details()
                input_shape = face_input_details[0]['shape']
                img_resized = cv2.resize(img_modified, (input_shape[2], input_shape[1]))
                if len(img_resized.shape) == 2:
                    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
                if face_input_details[0]['dtype'] == np.float32:
                    img_input = (img_resized.astype(np.float32) / 255.0)
                else:
                    img_input = img_resized.astype(np.uint8)
                img_input = np.expand_dims(img_input, axis=0)
                
                face_detector.set_tensor(face_input_details[0]['index'], img_input)
                face_detector.invoke()
                face_output = face_detector.get_tensor(face_detector.get_output_details()[0]['index'])
                
                st.warning(f"⚠️ Face Detection Score: {np.max(face_output):.4f}")
                st.info("This shows how brightness/contrast manipulation affects detection")
        
        elif attack_vector == "Blurring/Noise Addition":
            st.write("**Testing blurring and noise to bypass detection**")
            blur_amount = st.slider("Blur Kernel Size", 1, 31, 5, step=2)
            noise_level = st.slider("Noise Level (0-50)", 0, 50, 10)
            
            # Apply blur
            img_modified = cv2.GaussianBlur(img_array, (blur_amount, blur_amount), 0)
            # Add noise
            noise = np.random.normal(0, noise_level, img_modified.shape).astype(np.uint8)
            img_modified = cv2.add(img_modified, noise)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original")
            with col2:
                st.image(img_modified, caption="Modified (Blurred + Noise)")
            
            if st.button("Test Modified Image"):
                face_input_details = face_detector.get_input_details()
                input_shape = face_input_details[0]['shape']
                img_resized = cv2.resize(img_modified, (input_shape[2], input_shape[1]))
                if len(img_resized.shape) == 2:
                    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
                if face_input_details[0]['dtype'] == np.float32:
                    img_input = (img_resized.astype(np.float32) / 255.0)
                else:
                    img_input = img_resized.astype(np.uint8)
                img_input = np.expand_dims(img_input, axis=0)
                
                face_detector.set_tensor(face_input_details[0]['index'], img_input)
                face_detector.invoke()
                face_output = face_detector.get_tensor(face_detector.get_output_details()[0]['index'])
                
                st.warning(f"⚠️ Face Detection Score: {np.max(face_output):.4f}")
                st.info("Blur and noise can significantly degrade detection quality")
        
        else:
            st.info(f"Attack vector '{attack_vector}' testing interface - Coming soon!")

elif test_mode == "Batch Testing":
    st.subheader("📦 Batch Testing")
    
    st.write("Test multiple images at once")
    
    uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files and st.button("Run Batch Analysis"):
        results_list = []
        progress_bar = st.progress(0)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Face detection
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
            face_conf = float(np.max(face_output))
            
            # Anti-spoofing
            anti_spoof_input_details = anti_spoofing.get_input_details()
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
            anti_spoof_output = anti_spoofing.get_tensor(anti_spoofing.get_output_details()[0]['index'])
            spoof_score = float(np.max(anti_spoof_output))
            
            results_list.append({
                "filename": uploaded_file.name,
                "face_confidence": face_conf,
                "liveness_score": spoof_score,
                "verdict": "PASS" if (face_conf > confidence_threshold and spoof_score > spoofing_threshold) else "FAIL"
            })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        st.subheader("📊 Batch Results")
        
        # Display as table
        df_results = []
        for r in results_list:
            df_results.append({
                "File": r["filename"],
                "Face Conf": f"{r['face_confidence']:.4f}",
                "Liveness": f"{r['liveness_score']:.4f}",
                "Result": r["verdict"]
            })
        
        st.table(df_results)
        
        # Download results
        csv_data = "filename,face_confidence,liveness_score,verdict\n"
        for r in results_list:
            csv_data += f"{r['filename']},{r['face_confidence']:.6f},{r['liveness_score']:.6f},{r['verdict']}\n"
        
        st.download_button(
            label="Download Results (CSV)",
            data=csv_data,
            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

st.divider()
with st.sidebar:
    st.subheader("🔒 Security Research")
    st.markdown("""
    ### Testing Focus Areas:
    - Face detection bypass techniques
    - Anti-spoofing vulnerabilities
    - Model input manipulation
    - Edge cases and corner cases
    - Adversarial attacks
    
    ### Known Attack Vectors:
    - Printing and displaying photos
    - Mask/covering attacks
    - Brightness/contrast manipulation
    - Blur/motion attacks
    - Adversarial patches
    
    ### Model Analysis:
    - Input/output specifications
    - Confidence score distribution
    - Threshold sensitivity
    - Robustness testing
    """)
