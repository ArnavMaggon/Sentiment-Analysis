import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import speech_recognition as sr
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from deepface import DeepFace

# Download VADER Lexicon if not already downloaded
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()
recognizer = sr.Recognizer()

def analyze_sentiment():
    text = text_entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Warning", "Please enter some text to analyze!")
        return
    
    sentiment_score = sia.polarity_scores(text)
    compound_score = sentiment_score['compound']
    
    if compound_score >= 0.05:
        sentiment = "Positive"
    elif compound_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    result_label.config(text=f"Sentiment: {sentiment}\nScore: {compound_score}")

def analyze_facial_expression():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Could not capture image from camera.")
            break
        
        # Flip the frame for a mirrored view
        frame = cv2.flip(frame, 1)
        
        try:
            analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
            result_label.config(text=f"Detected Emotion: {emotion}")
        except Exception as e:
            result_label.config(text="Error in detecting emotion")
        
        cv2.putText(frame, f"Emotion: {emotion}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Facial Expression Analysis", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def analyze_voice_sentiment():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        messagebox.showinfo("Info", "Speak now...")
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio, language='en-US')  # Recognize English speech
            
            sentiment_score = sia.polarity_scores(text)
            compound_score = sentiment_score['compound']
            
            if compound_score >= 0.05:
                sentiment = "Positive"
            elif compound_score <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            result_label.config(text=f"Voice Sentiment: {sentiment}\nScore: {compound_score}")
        except sr.UnknownValueError:
            messagebox.showerror("Error", "Could not understand audio.")
        except sr.RequestError:
            messagebox.showerror("Error", "Could not request results from speech recognition service.")

# Create GUI application
app = tk.Tk()
app.title("Sentiment Analysis App")
app.geometry("400x400")

# UI Elements
label = tk.Label(app, text="Enter text for sentiment analysis:")
label.pack(pady=10)

text_entry = tk.Text(app, height=5, width=50)
text_entry.pack()

analyze_button = tk.Button(app, text="Analyze Sentiment", command=analyze_sentiment)
analyze_button.pack(pady=10)

facial_button = tk.Button(app, text="Analyze Facial Expression", command=analyze_facial_expression)
facial_button.pack(pady=10)

voice_button = tk.Button(app, text="Analyze Voice Sentiment", command=analyze_voice_sentiment)
voice_button.pack(pady=10)

result_label = tk.Label(app, text="", font=("Arial", 12, "bold"))
result_label.pack()

# Run application
app.mainloop()
