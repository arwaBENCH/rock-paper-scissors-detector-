# rock-paper-scissors-detector-
📌 Project Overview

This project implements a computer vision–based AI model capable of recognizing Rock, Paper, and Scissors gestures in real time using a webcam.

The model was trained on a custom dataset created from scratch, collected manually to better understand the full AI pipeline: from data acquisition to real-time inference.

The long-term goal of this project is to deploy the trained model on an STM microcontroller using TinyML, enabling edge AI inference without a PC.

🎯 Features

Real-time gesture recognition via webcam

Custom-built dataset + external Kaggle dataset

End-to-end ML pipeline

Ready for TinyML / embedded deployment

Lightweight model suitable for edge devices

🧠 Tech Stack

Programming Language: Python

Libraries:

OpenCV / TensorFlow / Keras / NumPy  

Hardware (current): PC + Webcam

Target Hardware (next): STM32 microcontroller


Classes:  Rock ✊ Paper ✋ Scissors ✌️


Creating the dataset manually helped ensure:

Better control over lighting and backgrounds

Deeper understanding of data bias and variability

⚙️ How It Works

Capture real-time video frames from the webcam

Preprocess frames (resize, normalize)

Feed frames into the trained model

Predict the gesture in real time

Display prediction on screen

🚀 Running the Project
git clone https://github.com/arwaBENCH/rock-paper-scissors-detector-
cd rock-paper-scissors-detector-
pip install -r requirements.txt
python main.py

Make sure your webcam is connected before running the script.

🔜 Future Work

Deploy on STM32 microcontroller

Interface with external camera module (e.g., OV7670)

Fully standalone embedded AI system



📬 Contact

If you’re interested in AI, embedded systems, or edge computing, feel free to connect!

LinkedIn: Arwa Ben Cheikh
