# 🤟 Real-Time Sign Language to Speech Translator

A beginner-friendly Python project that detects hand gestures using a webcam and translates basic sign language into speech in real-time using AI. Built using **OpenCV**, **MediaPipe**, and **pyttsx3**.

---

## 💡 Features

- 🔍 Detects hand movements and finger positions using webcam
- 🧠 Recognizes basic hand gestures (e.g., Fist = "Hello")
- 🗣️ Converts detected gesture into speech using Text-to-Speech
- 💻 Real-time video feed with gesture overlay

---

## 🧰 Tech Stack

| Tool         | Use                          |
|--------------|-------------------------------|
| Python       | Programming Language           |
| OpenCV       | Webcam feed & image processing |
| MediaPipe    | Hand landmark detection        |
| pyttsx3      | Offline text-to-speech         |

---

## 🚀 How to Run

### 1. Clone the repository or download the files
git clone https://github.com/arvind05kumar/Sign-Lang-Translator.git
cd Sign-Lang-Translator 

### 2. Install dependencies
pip install opencv-python mediapipe pyttsx3

### 3. Run the program
python sign_to_speech.py

### 4. Use hand gestures in front of the webcam
🫷 Make a fist to trigger "Hello" (you'll hear it via speaker)
