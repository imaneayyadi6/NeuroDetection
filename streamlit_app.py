import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd

IMG_SIZE = (150,150)

ALZ = ["MildDemented","ModerateDemented","NonDemented","VeryMildDemented"]
PRK_MRI = ["Normal","Parkinson"]
PRK_DRAW = ["Healthy","Parkinson"]

@st.cache_resource
def load_models():
    m1 = load_model("models/mon_modele_cnn.keras")
    m2 = load_model("models/mon_modele_cnn_prk.keras")
    m3 = load_model("models/mon_modele_cnn_handdraw.keras")
    return m1, m2, m3

def preprocess(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    x = np.array(img).astype("float32")/255.0
    return np.expand_dims(x, 0)

m_alz, m_prk, m_draw = load_models()

st.title("NeuroDetection (CNN)")

disease = st.sidebar.selectbox("Choose", ["Alzheimer", "Parkinson"])
test_type = None
if disease == "Parkinson":
    test_type = st.sidebar.radio("Test type", ["MRI", "Drawing"])

file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
if file:
    img = Image.open(file)
    st.image(img, use_container_width=True)
    x = preprocess(img)

    if st.button("Predict"):
        if disease == "Alzheimer":
            probs = m_alz.predict(x, verbose=0)[0]
            label = ALZ[int(np.argmax(probs))]
            classes = ALZ
        else:
            if test_type == "MRI":
                probs = m_prk.predict(x, verbose=0)[0]
                label = PRK_MRI[int(np.argmax(probs))]
                classes = PRK_MRI
            else:
                probs = m_draw.predict(x, verbose=0)[0]
                label = PRK_DRAW[int(np.argmax(probs))]
                classes = PRK_DRAW

        st.subheader(f"Prediction: {label}")
        df = pd.DataFrame({"class": classes, "prob": probs})
        st.bar_chart(df.set_index("class"))
