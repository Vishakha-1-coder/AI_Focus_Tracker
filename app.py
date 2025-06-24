import streamlit as st
import cv2
import time
import pygame
import numpy as np
from tensorflow.keras.models import load_model

# --- Constants ---
IMG_SIZE = 100
model = load_model("model/model.h5")

# --- Sound Alert Function ---
def play_alert(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

# --- Predict Focused/Distracted ---
def predict_focus(model, frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame.astype("float32") / 255.0
    input_data = np.expand_dims(frame, axis=0)
    pred = model.predict(input_data)[0]
    return np.argmax(pred) == 0  # Class 0 = focused

# --- Streamlit UI ---
st.set_page_config(page_title="AI Focus Tracker", layout="centered")
st.title("🧠 AI Focus Tracker")
st.header("Please keep the cammera at the side and start focusing.")
st.image("assets/hourglass.png", width=100)
focus_time = st.number_input("Enter focus time (in minutes)", min_value=1, step=1)
break_time = st.number_input("Enter break time (in minutes)", min_value=1, step=1)

col1, col2 = st.columns([1, 1])
start_btn = col1.button("▶️ Start Focus", use_container_width=True)
clear_btn = col2.button("🛑 Clear", use_container_width=True)

if clear_btn:
    st.warning("Session cleared. Ready for a new one.")

if start_btn:
    st.info("📸 Activating webcam...")
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    status = st.empty()
    timer_box = st.empty()

    start_time = time.time()
    distraction_alerted = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        focused = predict_focus(model, frame)

        stframe.image(frame, channels="BGR")

        elapsed = time.time() - start_time
        remaining = focus_time * 60 - elapsed
        mins, secs = divmod(int(remaining), 60)

        timer_box.markdown(
            f"""
            <div style="text-align:center; font-size:48px; font-weight:bold; color:#1f2937; margin-top:10px;">
                ⏳ {mins:02}:{secs:02}
            </div>
            """,
            unsafe_allow_html=True
        )

        if focused:
            status.success("✅ You are focused.")
            distraction_alerted = False
        else:
            status.warning("⚠️ Distraction detected!")
            if not distraction_alerted:
                play_alert("D:/AI_Focus_Tracker/assets/alert-sound-87478.wav")
                distraction_alerted = True

        if remaining <= 0:
            play_alert("D:/AI_Focus_Tracker/assets/alarm-327234.wav")
            st.balloons()
            st.success("🎉 Hurray! You completed your focus session.")
            break

        time.sleep(1)

    cap.release()
    st.info(f"🧘‍♀️ Time to take a break for {break_time} minutes!")
