# app.py
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import os

# # --- Configuration ---
# MODEL_PATH = "waste_classifier_v2.tflite"
# CLASS_NAMES = sorted([
#     'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 
#     'cardboard_boxes', 'cardboard_packaging', 'clothing', 'coffee_grounds', 
#     'disposable_plastic_cutlery', 'egg_shells', 'food_waste', 
#     'glass_beverage_bottles', 'glass_cosmetic_containers', 
#     'glass_food_jars', 'magazines', 'newspaper', 'office_paper', 'paper_cups', 
#     'plastic_cup_lids', 'plastic_detergent_bottles','plastic_food_containers', 'plastic_shopping_bags', 
#     'plastic_soda_bottles', 'plastic_straws', 'plastic_trash_bags', 
#     'plastic_water_bottles', 'shoes','steel_food_cans', 'styrofoam_cups','styrofoam_food_containers', 'tea_bags'
# ])

# # --- Prediction function ---
# def predict(input_image):
#     if input_image is None:
#         return None, None
    
#     # Load TFLite model
#     interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
#     interpreter.allocate_tensors()

#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     img_height = input_details[0]['shape'][1]
#     img_width = input_details[0]['shape'][2]

#     # Preprocess
#     image_resized = input_image.resize((img_width, img_height))
#     image_array = np.array(image_resized)
#     image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

#     # Predict
#     interpreter.set_tensor(input_details[0]['index'], image_array)
#     interpreter.invoke()

#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     prediction_probabilities = output_data[0]

#     top_idx = np.argmax(prediction_probabilities)
#     confidence = float(prediction_probabilities[top_idx])
#     return CLASS_NAMES[top_idx], confidence

# # --- Streamlit UI ---
# st.set_page_config(page_title="Garbage Detector", page_icon="🗑️", layout="wide")

# st.markdown("<h1 style='text-align:center;'>🗑️ Garbage Detection CNN Model</h1>", unsafe_allow_html=True)

# uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     if st.button("Classify Waste"):
#         label, confidence = predict(image)
#         if label:
#             st.success(f"**Prediction:** {label} ({confidence*100:.2f}%)")
#         else:
#             st.error("No prediction available.")



# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import os

# # --- Configuration ---
# MODEL_PATH = "models/waste_classifier_v2.tflite"
# CLASS_NAMES = sorted([
#     'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 
#     'cardboard_boxes', 'cardboard_packaging', 'clothing', 'coffee_grounds', 
#     'disposable_plastic_cutlery', 'egg_shells', 'food_waste', 
#     'glass_beverage_bottles', 'glass_cosmetic_containers', 
#     'glass_food_jars', 'magazines', 'newspaper', 'office_paper', 'paper_cups', 
#     'plastic_cup_lids', 'plastic_detergent_bottles','plastic_food_containers', 'plastic_shopping_bags', 
#     'plastic_soda_bottles', 'plastic_straws', 'plastic_trash_bags', 
#     'plastic_water_bottles', 'shoes','steel_food_cans', 'styrofoam_cups','styrofoam_food_containers', 'tea_bags'
# ])

# # --- Prediction function ---
# def predict(input_image):
#     if input_image is None:
#         return None, None
    
#     # Load TFLite model
#     interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
#     interpreter.allocate_tensors()

#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     img_height = input_details[0]['shape'][1]
#     img_width = input_details[0]['shape'][2]

#     # Preprocess
#     image_resized = input_image.resize((img_width, img_height))
#     image_array = np.array(image_resized)
#     image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

#     # Predict
#     interpreter.set_tensor(input_details[0]['index'], image_array)
#     interpreter.invoke()

#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     prediction_probabilities = output_data[0]

#     top_idx = np.argmax(prediction_probabilities)
#     confidence = float(prediction_probabilities[top_idx])
#     return CLASS_NAMES[top_idx], confidence

# # --- Streamlit UI ---
# st.set_page_config(page_title="Garbage Detector", page_icon="🗑", layout="wide")

# st.markdown("<h1 style='text-align:center;'>🗑 Garbage Detection CNN Model</h1>", unsafe_allow_html=True)

# uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     if st.button("Classify Waste"):
#         label, confidence = predict(image)
#         if label:
#             st.success(f"Prediction: {label} ({confidence*100:.2f}%)")
#         else:
#             st.error("No prediction available.")


# app.py
# import streamlit as st
# from PIL import Image
# from predictor import predict # Import the prediction function

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Garbage Detector",
#     page_icon="🗑",
#     layout="centered",
#     initial_sidebar_state="auto"
# )

# # --- UI Components ---
# st.title("🗑 Garbage Detection CNN Model")
# st.write(
#     "Upload an image or use your camera to classify a piece of waste. "
#     "The model will predict which of the 30 categories it belongs to."
# )

# # Create two columns for different input methods
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("📤 Upload an Image")
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# with col2:
#     st.subheader("📸 Use Your Camera")
#     camera_file = st.camera_input("Take a picture")

# # --- Prediction Logic ---
# # Determine which input has been provided
# image_to_process = None
# if uploaded_file is not None:
#     image_to_process = uploaded_file
# elif camera_file is not None:
#     image_to_process = camera_file

# # If an image is available, process and predict
# if image_to_process is not None:
#     # Display the image
#     image = Image.open(image_to_process)
#     st.image(image, caption="Your Image", use_column_width=True)
    
#     # Make a prediction
#     st.write("Classifying...")
#     label, confidence = predict(image)
    
#     # Display the result
#     st.success(f"*Prediction:* {label}")
#     st.info(f"*Confidence:* {confidence*100:.2f}%")

#     # Optional: Display a confidence bar
#     st.progress(confidence)
# else:
#     st.info("Please upload an image or take a picture to get a prediction.")
    
    


import streamlit as st
from PIL import Image
from predictor import predict

# ------------------- Page Config -------------------
st.set_page_config(page_title="Garbage Detector", page_icon="🗑", layout="wide")

# ------------------- Session State -------------------
if "step" not in st.session_state:
    st.session_state.step = 1

def go_to_camera():
    st.session_state.step = 2

def go_to_prediction():
    st.session_state.step = 3

# ------------------- CSS: Stunning UI -------------------
st.markdown("""
<style>
/* Import Fonts */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap');

/* Animated Gradient Background with Floating Shapes */
body {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(-45deg, #d1f8ff, #ffe7f0, #fff4d9, #e1ffe4);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    overflow-x: hidden;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Floating Particles (Subtle) */
body::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: radial-gradient(rgba(255,255,255,0.3) 1px, transparent 1px);
    background-size: 40px 40px;
    animation: moveParticles 8s linear infinite;
}
@keyframes moveParticles {
    0% { background-position: 0 0; }
    100% { background-position: 40px 40px; }
}

/* Title with Neon Glow + Float */
.title {
    font-size: 3.5rem;
    font-weight: 700;
    text-align: center;
    color: #2c3e50;
    text-shadow: 0 0 15px rgba(0,0,0,0.2);
    animation: float 3s ease-in-out infinite alternate, fadeIn 1.2s ease-in-out;
}
@keyframes float {
    0% { transform: translateY(0px); }
    100% { transform: translateY(-8px); }
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #555;
    margin-bottom: 2rem;
    animation: fadeIn 1.5s ease-in-out;
}

/* Card (Upload/Camera) with Glassmorphism */
.upload-box {
    background: rgba(255, 255, 255, 0.5);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.3);
    text-align: center;
    transition: transform 0.4s ease, box-shadow 0.4s ease;
    animation: slideUp 0.8s ease forwards;
}
.upload-box:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 12px 40px rgba(0,0,0,0.2);
}
@keyframes slideUp {
    0% { transform: translateY(40px); opacity: 0;}
    100% { transform: translateY(0); opacity: 1;}
}

/* Buttons - Elegant Gradient with Shine */
div.stButton > button {
    background: linear-gradient(90deg, #74ebd5, #9face6);
    color: #fff;
    border: none;
    padding: 0.8rem 1.8rem;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
div.stButton > button::after {
    content: '';
    position: absolute;
    top: 0; left: -50%;
    width: 200%; height: 100%;
    background: rgba(255,255,255,0.3);
    transform: skewX(-20deg);
    transition: left 0.5s;
}
div.stButton > button:hover::after {
    left: 100%;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

/* Loader - AI Pulse Radar Style */
.loader {
    width: 60px; height: 60px;
    border-radius: 50%;
    background: radial-gradient(circle, #74ebd5 10%, transparent 70%);
    animation: pulse 1.5s infinite ease-in-out;
    margin: 30px auto;
}
@keyframes pulse {
    0% { transform: scale(0.9); opacity: 0.7;}
    50% { transform: scale(1.2); opacity: 1;}
    100% { transform: scale(0.9); opacity: 0.7;}
}

/* Result Box - Gradient Border Glow */
.result-box {
    background: rgba(255,255,255,0.7);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    margin-top: 1.5rem;
    font-size: 1.4rem;
    font-weight: 600;
    border: 3px solid transparent;
    background-clip: padding-box;
    box-shadow: 0px 4px 25px rgba(0,0,0,0.15);
    animation: fadeIn 1s ease forwards, glow 2s infinite alternate;
}
@keyframes glow {
    0% { box-shadow: 0 0 10px #74ebd5; }
    100% { box-shadow: 0 0 20px #9face6; }
}

/* Image Hover Zoom */
img {
    border-radius: 12px;
    transition: transform 0.4s ease, box-shadow 0.4s ease;
}
img:hover {
    transform: scale(1.03);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# ------------------- Title & Subtitle -------------------
st.markdown("<h1 class='title'>🗑 Garbage Detection CNN Model</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload or capture an image to classify waste into 30 AI-powered categories.</p>", unsafe_allow_html=True)

# ------------------- Step Routing -------------------
if st.session_state.step == 1:
    # Upload & Camera Option
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        st.subheader("📤 Upload an Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        st.markdown("</div>", unsafe_allow_html=True)
        if uploaded_file:
            st.session_state.uploaded = uploaded_file
            st.session_state.step = 3

    with col2:
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        st.subheader("📸 Take a Picture")
        st.button("Open Camera", on_click=go_to_camera)
        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.step == 2:
    # Camera Input
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    st.subheader("📸 Take a Picture")
    camera_file = st.camera_input("Capture your waste image")
    st.markdown("</div>", unsafe_allow_html=True)
    if camera_file:
        st.session_state.uploaded = camera_file
        st.session_state.step = 3

if st.session_state.step == 3 and "uploaded" in st.session_state:
    # Prediction Step
    image = Image.open(st.session_state.uploaded)
    st.image(image, caption="Your Image", use_column_width=True)

    st.markdown("<div class='loader'></div>", unsafe_allow_html=True)
    label, confidence = predict(image)

    # Dynamic Icon
    icons = {"plastic":"♻️","organic":"🌱","metal":"⚙️","paper":"📄","glass":"🪟"}
    icon = icons.get(label.lower(),"🗑")

    st.markdown(f"""
        <div class="result-box">
            {icon} Prediction: <b>{label}</b><br>
            Confidence: <b>{confidence*100:.2f}%</b>
        </div>
    """, unsafe_allow_html=True)
    st.progress(confidence)

    st.button("Restart", on_click=lambda: st.session_state.update(step=1, uploaded=None))
