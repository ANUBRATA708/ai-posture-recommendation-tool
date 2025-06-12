import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Streamlit config
st.set_page_config(page_title="🧍 AI Posture Advisor", page_icon="🧘", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .title {
            font-size:36px !important;
            color:#4e73df;
            text-align: center;
        }
        .recommend-box {
            border: 2px dashed #4e73df;
            background-color: #f0f8ff;
            padding: 20px;
            border-radius: 15px;
        }
        .good {
            color: green;
            font-size: 20px;
            font-weight: bold;
        }
        .bad {
            color: red;
            font-size: 20px;
            font-weight: bold;
        }
        .disclaimer {
            font-size: 12px;
            color: gray;
            text-align: center;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("posture_model.h5")

model = load_model()

# Helper functions
def preprocess_image(img):
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return img

def predict_posture(image_array):
    img = preprocess_image(image_array)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    return "Good Posture" if prediction > 0.5 else "Bad Posture", prediction

def get_recommendation(pred):
    if pred <= 0.5:
        return (
            "⚠️ **Bad Posture Detected!**\n\n"
            "**Recommendations:**\n"
            "- Sit upright with your back straight\n"
            "- Keep your shoulders relaxed\n"
            "- Feet flat on the floor\n"
            "- Screen at eye level\n"
            "- Take short breaks every 30 minutes"
        )
    else:
        return "✅ You're maintaining a healthy posture! Keep it up!"

# UI
st.markdown('<div class="title">🧍 AI-Assisted Posture Advisor</div>', unsafe_allow_html=True)
st.markdown("### 📸 Detect your sitting posture and receive personalized tips!")

tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Use Webcam"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_np = np.array(image)
        result, prob = predict_posture(image_np)

        if result:
            color_class = "good" if result == "Good Posture" else "bad"
            st.markdown(f'<div class="{color_class}">🧠 Prediction: {result} ({prob:.2f})</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="recommend-box">{get_recommendation(prob)}</div>', unsafe_allow_html=True)

with tab2:
    capture = st.button("📷 Capture from Webcam")

    if capture:
        cap = cv2.VideoCapture(0)
        st.warning("🛑 Press 'Q' in the popup window to capture and close webcam.")
        captured = False

        while not captured:
            ret, frame = cap.read()
            if not ret:
                st.error("Webcam access failed.")
                break
            cv2.imshow("Press Q to Capture Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                captured = True
                cap.release()
                cv2.destroyAllWindows()

                st.image(frame, caption="Captured Image", channels="BGR", use_column_width=True)

                result, prob = predict_posture(frame)
                if result:
                    color_class = "good" if result == "Good Posture" else "bad"
                    st.markdown(f'<div class="{color_class}">🧠 Prediction: {result} ({prob:.2f})</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="recommend-box">{get_recommendation(prob)}</div>', unsafe_allow_html=True)

# Privacy notice
st.markdown('<div class="disclaimer">🔒 Your webcam images are processed in-memory only and never stored or shared.</div>', unsafe_allow_html=True)
