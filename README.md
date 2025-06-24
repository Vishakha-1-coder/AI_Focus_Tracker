# 🧠 AI Focus Tracker

A real-time AI-powered desktop app that monitors user focus via webcam. If distraction is detected (e.g., looking away, closing eyes), it plays an alert sound. This project is built with **Streamlit**, **OpenCV**, and a custom-trained **Convolutional Neural Network (CNN)** model.

---

## 🔍 Features

- 📸 **Webcam-Based Monitoring**  
  Uses real-time webcam feed to monitor your facial focus.

- ⚠️ **Distraction Detection**  
  Alerts you when you're distracted using a trained machine learning model.

- ⏳ **Session Timer**  
  Set your focus and break durations just like a Pomodoro timer.

- 🔊 **Audio Alerts**  
  Get notified via sound alerts when distracted or when time is up.

---

## 🧠 AI Model

A CNN model was trained on the [State Farm Distracted Driver Dataset](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data), where:
- `c0` (driver focused) is labeled as `focused`
- `c1` to `c9` (other activities like texting, drinking, etc.) are labeled as `distracted`

Model Architecture:
- Conv2D + MaxPooling Layers
- Fully Connected Dense Layers
- Softmax output for binary classification (`focused` vs `distracted`)

---

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/Vishakha-1-coder/AI_Focus_Tracker.git
cd AI_Focus_Tracker

2. **Install dependencies**
```bash
pip install -r requirements.txt

3. **Start the app**
```bash
streamlit run app.py

🧠 Use Cases
Students who want to improve focus while studying

Remote workers trying to reduce screen fatigue

Productivity hackers applying Pomodoro techniques

🙌 Credits
Model trained on: State Farm Distracted Driver Dataset

Built with ❤️ by Vishakha Karande


