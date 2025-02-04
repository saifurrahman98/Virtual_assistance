import os
import numpy as np
import speech_recognition as sr
import pyttsx3
import requests
import music_library
import webbrowser
from openai import OpenAI
from gtts import gTTS
import pygame
import soundfile as sf
from python_speech_features import mfcc


# Initialize recognizer and text-to-speech engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()
newsapi = 'a1d9a008ac2549d5b91ad89c5a93912b'
REFERENCE_VOICE_PATH = "reference_voice_new.wav"

def speak(text):
    tts = gTTS(text)
    tts.save('temp.mp3')
    
    # Initialize Pygame mixer
    pygame.mixer.init()
 
    # Load the MP3 file
    pygame.mixer.music.load('temp.mp3')

    # Play the MP3 file
    pygame.mixer.music.play()

    # Keep the program running until the music stops playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    os.remove("temp.mp3")

def aiprocess(command):
    client = OpenAI(api_key="sk-proj-B1V5vyIV3a3iR1QpsyZkvQRC-vScvQQSvcig6Sj5U9qHY8zybFRKJVBKblRsqD5gnoTL_XW19T3BlbkFJ8OGjV_1PS8pZn7GmdQza2DAAPBIobhiCmbVpSPLidcWhfFS7O4zsgVMJUjgh_EYLPiY1-_gjYA",)
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a virtual assistant named Marcus skilled in general tasks like Alexa and google cloud.give short answer"},
        { "role": "user", "content": command}
    ]
    )

    return (completion.choices[0].message.content)

def processCommand(c):
    if "open google" in c.lower():
          webbrowser.open("https://google.com")
    elif "open youtube" in c.lower():
          webbrowser.open("https://youtube.com")
    elif "open facebook" in c.lower():
          webbrowser.open("https://facebook.com") 
    elif "open linkedin" in c. lower():
          webbrowser.open ("https://linkedin.com")
    elif c.lower().startswith("play"):
         song = c.lower().split(" ")[1]
         link = music_library.music[song]
         webbrowser.open(link)
    elif "news" in c.lower():
         r=requests.get(f'https://newsapi.org/v2/top-headlines?country=us&apiKey={newsapi}')
         if r.status_code == 200:
            data = r.json()                       #Parse the JSON response
            articles = data.get('articles',[])    #Extract the articles
            for article in articles:           
                print(article["title"])
                speak(article["title"])
    else:
         #lets handled to open AI
         output=aiprocess(c)
         speak(output)

def capture_reference_voice():
    """Record and save the reference voice for verification."""
    with sr.Microphone() as source:
        print("Please say 'Marcus' to set up your voice sample.")
        audio = recognizer.listen(source)
        with open(REFERENCE_VOICE_PATH, "wb") as f:
            f.write(audio.get_wav_data())
    print("Reference voice saved.")
    
def load_audio(file_path):
    """Load audio file using soundfile (alternative to librosa)."""
    audio, sample_rate = sf.read(file_path)  # Use soundfile to read the audio file
    return audio, sample_rate

def compute_mfcc(audio, sample_rate, fixed_length=2.0):
    """
    Compute MFCC features for fixed-length audio.
    - audio: The audio data as a NumPy array.
    - sample_rate: The sampling rate of the audio.
    - fixed_length: Desired duration of audio in seconds.
    """
    # Ensure audio is of fixed length
    target_length = int(fixed_length * sample_rate)
    if len(audio) > target_length:
        audio = audio[:target_length]  # Truncate
    else:
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')  # Pad with zeros

    # Compute MFCC features with a higher NFFT
    mfcc_features = mfcc(audio, samplerate=sample_rate, numcep=20, nfft=2048)  # Use 20 cepstral coefficients
    return mfcc_features

def verify_voice(input_audio_path, reference_audio_path, fixed_length=2.0):
    """Compare input voice with reference voice."""
    # Load the reference and input audio
    ref_audio, ref_sr = load_audio(reference_audio_path)
    input_audio, input_sr = load_audio(input_audio_path)

    # Extract MFCC features with fixed length
    ref_mfcc = compute_mfcc(ref_audio, ref_sr, fixed_length)
    input_mfcc = compute_mfcc(input_audio, input_sr, fixed_length)

    # Print shapes to check alignment
    print(f"Reference MFCC Shape: {ref_mfcc.shape}")
    print(f"Input MFCC Shape: {input_mfcc.shape}")

    # Compute similarity using cosine similarity
    ref_mfcc_flatten = ref_mfcc.flatten()
    input_mfcc_flatten = input_mfcc.flatten()
    similarity = np.dot(ref_mfcc_flatten, input_mfcc_flatten) / (
        np.linalg.norm(ref_mfcc_flatten) * np.linalg.norm(input_mfcc_flatten)
    )
    
    # Log similarity score
    print(f"Cosine Similarity: {similarity}")

    return similarity > 0.8 # Increased threshold to improve accuracy

if __name__ == "__main__":
    # Set up reference voice if not already saved
    if not os.path.exists(REFERENCE_VOICE_PATH):
        capture_reference_voice()

    speak("Initializing virtual assistant")

    while True:
        # Listen for the wake word "MARCUS"
        try:
            with sr.Microphone() as source:
                print("Listening...")
                audio = recognizer.listen(source, timeout=3)  # Adjust timeout limits
            word = recognizer.recognize_google(audio)  # Use Google's recognizer

            if word.lower() in ["marcus", "hey marcus"]:
                # Save the input voice for verification
                input_voice_path = "input_voice.wav"
                with open(input_voice_path, "wb") as f:
                    f.write(audio.get_wav_data())
                
                # Verify the voice
                if verify_voice(input_voice_path, REFERENCE_VOICE_PATH):
                    speak("Voice verified. How can I assist you?")
                    os.remove(input_voice_path)

                    # Listen for further commands
                    with sr.Microphone() as source:
                        print("Marcus active...")
                        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                        command = recognizer.recognize_google(audio)
                        print(f"Command: {command}")
                        processCommand(command)

                    if "stop" in command.lower():
                        speak("Shutting down... Goodbye")
                        break
                else:
                    speak("Voice not recognized. Access denied.")
                    os.remove(input_voice_path)

        except Exception as e:
            print("Error: ", format(e))