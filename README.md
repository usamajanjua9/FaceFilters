# Webcam Face Detection & Filters App

This Streamlit application provides **real-time face detection** using a webcam, allows users to apply **various filters**, and supports **overlaying custom glasses images** on detected faces.

[Run](https://facefilters-kzb59k48rotapp7fddhmpyz.streamlit.app/)!

## Features

### 🔍 Face Detection
- Uses OpenCV's **Haarcascade** to detect faces in real-time.
- Draws **bounding boxes** around detected faces.

### 🎨 Filters
- Users can choose from **10+ filters**, including:
  - Grayscale
  - Cartoon Effect
  - Blur
  - Edge Detection
  - Pencil Sketch
  - Sepia
  - Invert Colors
  - Emboss
  - Sharpen
  - HSV

### 🕶 Glasses Overlay
- Users can **upload a PNG image of glasses**, and the app will **automatically place them** on detected faces.

### 📷 Webcam Controls
- Start/Stop **live webcam feed**.
- Take **snapshots** of the processed frame.

## Installation

To run this app, install dependencies using:

```sh
pip install -r requirements.txt
```

## Running the Application

Run the following command:

```sh
streamlit run streamlit_face_filters.py
```

## Developed By
🚀 **Dr. Usama Arshad**

