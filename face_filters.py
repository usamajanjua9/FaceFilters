import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Set up Streamlit UI
st.set_page_config(page_title="Webcam Face Detection & Filters", page_icon="📷", layout="wide")
st.title("📷 Webcam Face Detection, Filters, Emojis & Glasses Overlay 🎭")

# Hide Streamlit branding
st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    .viewerBadge_container__1QSob {display: none !important;}
    .reportview-container .main .block-container {padding-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True
)

# Load Haarcascade for Face Detection
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Sidebar Options
st.sidebar.header("🔧 Options")
filter_option = st.sidebar.radio("🎨 Choose a Filter:", 
                                 ("No Filter", "Grayscale", "Cartoon", "Blur", 
                                  "Edge Detection", "Pencil Sketch", "Sepia", 
                                  "Invert Colors", "Emboss", "Sharpen", "HSV"))

# Add sliders for intensity and brightness control
filter_intensity = st.sidebar.slider("🎛 Filter Intensity", 1, 10, 5)
brightness_factor = st.sidebar.slider("💡 Brightness", -100, 100, 0)

# Glasses and Emoji options
glasses_image = st.sidebar.file_uploader("🕶 Upload Glasses Image (PNG with Transparency)", type=["png"])
emoji_option = st.sidebar.selectbox("😃 Choose an Emoji to Overlay:", 
                                    ("None", "😂", "😎", "😍", "🤩", "👽", "🐱"))

# Image Upload Option
st.sidebar.subheader("📷 Upload an Image")
uploaded_image = st.sidebar.file_uploader("📂 Choose an image", type=["jpg", "jpeg", "png"])

# Webcam Capture Option
st.sidebar.subheader("📸 Take a Picture")
image_file = st.camera_input("Capture an Image")

# Video Upload Option
st.sidebar.subheader("📹 Upload a Video")
video_file = st.sidebar.file_uploader("📂 Choose a video", type=["mp4", "avi", "mov"])

# Function to get a valid emoji-supporting font
def get_emoji_font():
    possible_fonts = [
        "/usr/share/fonts/noto/NotoColorEmoji.ttf",  # Linux
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",  # Linux (alternate)
        "/System/Library/Fonts/Supplemental/AppleColorEmoji.ttc",  # macOS
        "C:/Windows/Fonts/seguiemj.ttf",  # Windows
    ]

    for font_path in possible_fonts:
        if os.path.exists(font_path):
            return font_path

    st.warning("⚠ Could not find an emoji-compatible font. Emojis may not render properly.")
    return None  # No valid font found

# Function to apply image filters
def apply_filter(frame, filter_option, intensity):
    if filter_option == "Grayscale":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Fix single-channel issue

    elif filter_option == "Cartoon":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        gray = cv2.medianBlur(gray, intensity)  # Fix for medianBlur()
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, intensity * 2, 250, 250)
        frame = cv2.bitwise_and(color, color, mask=edges)

    elif filter_option == "Blur":
        frame = cv2.GaussianBlur(frame, (intensity * 2 + 1, intensity * 2 + 1), 0)

    elif filter_option == "Edge Detection":
        frame = cv2.Canny(frame, intensity * 10, intensity * 20)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Fix single-channel issue

    elif filter_option == "Pencil Sketch":
        try:
            _, sketch = cv2.pencilSketch(frame, sigma_s=intensity * 10, sigma_r=0.07, shade_factor=0.05)
            frame = sketch
        except AttributeError:
            st.warning("⚠ Pencil Sketch filter is not available in your OpenCV version.")
            return frame

    elif filter_option == "Sepia":
        kernel = np.array([[0.272, 0.534, 0.131], 
                           [0.349, 0.686, 0.168], 
                           [0.393, 0.769, 0.189]])
        frame = cv2.transform(frame, kernel * (intensity / 10))

    elif filter_option == "Invert Colors":
        frame = cv2.bitwise_not(frame)

    elif filter_option == "Emboss":
        kernel = np.array([[0, -intensity, -intensity], 
                           [intensity, 0, -intensity], 
                           [intensity, intensity, 0]])
        frame = cv2.filter2D(frame, -1, kernel)

    elif filter_option == "Sharpen":
        kernel = np.array([[0, -intensity, 0], 
                           [-intensity, intensity * 5, -intensity], 
                           [0, -intensity, 0]])
        frame = cv2.filter2D(frame, -1, kernel)

    elif filter_option == "HSV":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)  # Convert back to BGR

    return frame

# Function to overlay emoji using PIL
def overlay_emoji(frame, emoji_option):
    if emoji_option == "None":
        return frame

    # Convert OpenCV image to PIL format
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_frame)

    # Get emoji-compatible font
    font_path = get_emoji_font()
    if font_path:
        font = ImageFont.truetype(font_path, 60)  # Adjust size
    else:
        return frame  # No font available, return unmodified image

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        draw.text((x + int(w / 3), y + int(h / 2)), emoji_option, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

# Process Uploaded or Captured Image
if uploaded_image or image_file:
    image = Image.open(uploaded_image) if uploaded_image else Image.open(image_file)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    frame = apply_filter(frame, filter_option, filter_intensity)
    frame = overlay_emoji(frame, emoji_option)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame, channels="RGB", caption="Processed Image 🎭")
    st.success("📸 Image processed successfully!")

st.sidebar.info("🚀 Developed by Dr. Usama Arshad")
