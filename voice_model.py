import os
import torch
from speechbrain.pretrained import SpeakerRecognition
from speechbrain.dataio.dataio import read_audio
import pyaudio

# Function to train and extract embeddings for your voice
def extract_embeddings_for_speaker(audio_folder):
    speaker = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="tmpdir")

    # List all audio files in the folder
    audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith(".wav")]
    embeddings = []

    # Extract embeddings from each file
    for file in audio_files:
        signal, fs = read_audio(file)
        emb = speaker.encode_batch([signal])  # Getting embeddings
        embeddings.append(emb)

    # Save the embeddings of your voice to compare in real-time
    torch.save(embeddings, "your_voice_embeddings.pt")
    print("Embeddings saved!")

# Function to recognize live audio from the microphone and compare with saved embeddings
def recognize_live_audio():
    print("Please speak to detect your voice...")

    # Load saved embeddings
    saved_embeddings = torch.load("your_voice_embeddings.pt")

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Set audio parameters
    RATE = 16000  # Sample rate
    CHUNK = 1024  # Size of each audio chunk
    FORMAT = pyaudio.paInt16  # Audio format
    CHANNELS = 1  # Single-channel audio (mono)

    # Initialize SpeakerRecognition model
    speaker = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="tmpdir")
    
    # Start the stream to listen for your voice
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    try:
        while True:
            audio_chunk = stream.read(CHUNK)
            
            # Extract the embedding for the captured audio chunk
            signal, fs = read_audio(audio_chunk)
            embedding = speaker.encode_batch([signal])

            # Compare the embeddings
            similarity = torch.cosine_similarity(saved_embeddings[0], embedding)  # Compare with saved embeddings

            if similarity > 0.8:  # You can adjust this threshold
                print("Speaker recognized as YOU!")
            else:
                print("Speaker not recognized!")
                
    except KeyboardInterrupt:
        print("\nExiting...")
        stream.stop_stream()
        stream.close()
        p.terminate()

# Main process
if __name__ == "__main__":
    audio_folder = "your_voice_folder"  # Path to your folder with voice samples

    # Step 1: Extract embeddings for your voice
    extract_embeddings_for_speaker(audio_folder)

    # Step 2: Start live audio recognition
    recognize_live_audio()






# import os
# from pyAudioAnalysis import audioTrainTest as aT
# import speech_recognition as sr

# REFERENCE_VOICE_PATH = "/Users/saifurrahman/Desktop/Virtual_Assistant_project/voices"  # Folder with multiple .wav files of your voice

# def train_model(REFERENCE_VOICE_PATH):
#     model_name="voice_model"
#     # Define parameters for feature extraction and training
#     mid_step = 1.0  # Middle step for feature extraction
#     short_window = 0.05  # Short window size (in seconds)
#     short_step = 0.025  # Short step size (in seconds)
#     classifier_type = "svm"  # Classifier type (e.g., "svm", "knn", "randomforest")
    
#     # Get all .wav files in the folder
#     wav_files = [os.path.join(REFERENCE_VOICE_PATH, f) for f in os.listdir(REFERENCE_VOICE_PATH) if f.endswith(".wav")]

#     # Check if there are any .wav files in the folder
#     if not wav_files:
#         print(f"No .wav files found in {REFERENCE_VOICE_PATH}")
#         return

#     # Extract features from the reference audio files and train the model
#     aT.extract_features_and_train(wav_files, mid_step, short_window, short_step, classifier_type, model_name)
#     print(f"Model training complete and saved as {model_name}.xml")

# def recognize_voice(test_audio_file, model_name="voice_model"):
#     # Classify the test audio using the trained model
#     result, _ = aT.file_classification(test_audio_file, model_name + ".xml")
    
#     # Assuming '1' corresponds to your voice in the model
#     if result == 1:
#         print("Voice recognized as yours!")
#     else:
#         print("Voice not recognized.")

# def record_and_test_your_voice():
#     model_name="voice_model"
#     recognizer = sr.Recognizer()
#     mic = sr.Microphone()

#     with mic as source:
#         print("Say something...")
#         audio = recognizer.listen(source)
#         print("Processing your speech...")

#         try:
#             # Save the recorded audio to a temporary file
#             with open("test.wav", "wb") as f:
#                 f.write(audio.get_wav_data())

#             # Recognize the voice from the test audio
#             recognize_voice("test.wav", model_name)
#         except Exception as e:
#             print(f"Error: {e}")

# # Train the model with the reference voice samples in the folder
# train_model("/Users/saifurrahman/Desktop/Virtual_Assistant_project/voices")

# # Now, you can test your voice recognition
# record_and_test_your_voice()



# def extract_features(filename):
#     # Load the audio file
#     audio, sr = librosa.load(filename, sr=None)
#     # Extract MFCC features
#     mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
#     # Return the mean of the MFCCs
#     return np.mean(mfccs, axis=1)

# def compare_voice(test_voice, reference_voice):
#     test_features = extract_features(test_voice)
#     reference_features = extract_features(reference_voice)
    
#     # Calculate Euclidean distance between features
#     distance = np.linalg.norm(test_features - reference_features)
    
#     return distance

# def recognize_your_voice(reference_voice_file, threshold=10.0):
#     recognizer = sr.Recognizer()
#     mic = sr.Microphone()

#     with mic as source:
#         print("Say something...")
#         audio = recognizer.listen(source)
#         print("Processing your speech...")

#         try:
#             # Save the recorded audio to a temporary file
#             with open("test.wav", "wb") as f:
#                 f.write(audio.get_wav_data())

#             # Compare the recorded audio with your reference voice
#             distance = compare_voice("test.wav", reference_voice_file)
#             print(f"Distance between voices: {distance}")

#             if distance < threshold:
#                 print("Voice recognized as yours!")
#             else:
#                 print("Voice not recognized.")
#         except Exception as e:
#             print(f"Error: {e}")

# # Replace 'your_voice_sample.wav' with the path to your recorded voice file
# recognize_your_voice("your_voice_sample.wav")