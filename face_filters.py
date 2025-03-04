import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Customizing Streamlit UI
st.set_page_config(page_title="Webcam Face Detection & Filters", page_icon="📷", layout="wide")
st.title("📷 Webcam Face Detection, Filters & Glasses Overlay")

# Load Haarcascade for Face Detection
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Sidebar Options
st.sidebar.header("🔧 Options")
filter_option = st.sidebar.radio("Choose a Filter:", ("No Filter", "Grayscale", "Cartoon", "Blur", "Edge Detection", "Pencil Sketch", "Sepia", "Invert Colors", "Emboss", "Sharpen", "HSV"))

glasses_image = st.sidebar.file_uploader("Upload Glasses Image (PNG with Transparency)", type=["png"])

def overlay_glasses(frame, glasses):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 4)
    
    if glasses is not None:
        glasses = cv2.cvtColor(np.array(glasses), cv2.COLOR_RGBA2BGRA)
        for (x, y, w, h) in faces:
            gw, gh = int(w * 0.9), int(h * 0.3)
            gx, gy = x + int(w * 0.05), y + int(h * 0.2)
            glasses_resized = cv2.resize(glasses, (gw, gh))
            
            for i in range(gh):
                for j in range(gw):
                    if glasses_resized[i, j, 3] > 0:  # Only overlay non-transparent parts
                        frame[gy + i, gx + j] = glasses_resized[i, j, :3]
    
    return frame

def apply_filter(frame, filter_option):
    if filter_option == "Grayscale":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif filter_option == "Cartoon":
        gray = cv2.medianBlur(frame, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, 9, 250, 250)
        frame = cv2.bitwise_and(color, color, mask=edges)
    elif filter_option == "Blur":
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
    elif filter_option == "Edge Detection":
        frame = cv2.Canny(frame, 100, 200)
    elif filter_option == "Pencil Sketch":
        _, sketch = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
        frame = sketch
    elif filter_option == "Sepia":
        kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
        frame = cv2.transform(frame, kernel)
    elif filter_option == "Invert Colors":
        frame = cv2.bitwise_not(frame)
    elif filter_option == "Emboss":
        kernel = np.array([[0,-1,-1],[1,0,-1],[1,1,0]])
        frame = cv2.filter2D(frame, -1, kernel)
    elif filter_option == "Sharpen":
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        frame = cv2.filter2D(frame, -1, kernel)
    elif filter_option == "HSV":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return frame

# Webcam Image Input (for Streamlit Cloud)
st.sidebar.subheader("📷 Capture an Image")
image_file = st.camera_input("Take a picture")

if image_file is not None:
    image = Image.open(image_file)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    frame = apply_filter(frame, filter_option)
    
    if glasses_image is not None:
        glasses = Image.open(glasses_image)
        frame = overlay_glasses(frame, glasses)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame, channels="RGB", caption="Processed Image")
    st.success("📸 Image processed successfully!")

st.sidebar.info("🚀 Developed by Dr. Usama Arshad")
