import speech_recognition as sr
import os
import time

# Path where the audio files will be stored
REFERENCE_VOICE_PATH = "/Users/saifurrahman/Desktop/my_projects/Virtual_Assistant_project/voices/user_voice"

# Create the folder if it doesn't exist
if not os.path.exists(REFERENCE_VOICE_PATH):
    os.makedirs(REFERENCE_VOICE_PATH)

# Initialize recognizer
recognizer = sr.Recognizer()

# Prompt user to say a specific phrase to record their voice
with sr.Microphone() as source:
    print("Please say 'Marcus' to set up your voice sample.")
    audio = recognizer.listen(source)
    
    # Generate a unique filename using the current timestamp
    file_name = os.path.join(REFERENCE_VOICE_PATH, f"sample{int(time.time())}.wav")
    
    # Save the audio to the specified path with a .wav extension
    with open(file_name, "wb") as f:
        f.write(audio.get_wav_data())
    
    print(f"Reference voice saved as {file_name}")