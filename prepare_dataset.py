import os
import whisper
from pydub import AudioSegment

# Load Whisper model
model = whisper.load_model("base")  # You can use "small", "medium", or "large" based on your preference

# Path to your dataset
audio_dir = "/home/jupyter/advanced/asr/audio/"
transcription_dir = "/home/jupyter/advanced/asr/transcriptions/"

# Function to load and preprocess audio
def load_and_preprocess_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    temp_wav_path = "temp_audio.wav"
    audio.export(temp_wav_path, format="wav")
    return temp_wav_path

# Function to transcribe a single audio file
def transcribe_audio(audio_path):
    processed_audio = load_and_preprocess_audio(audio_path)
    result = model.transcribe(processed_audio)
    os.remove(processed_audio)  # Clean up the temporary file
    return result["text"]

# Process all audio files in the dataset
def transcribe_dataset():
    transcriptions = {}
    
    # Iterate through all audio files in the directory
    for audio_file in os.listdir(audio_dir):
        if audio_file.endswith(".wav") or audio_file.endswith(".mp3"):
            audio_path = os.path.join(audio_dir, audio_file)
            transcription = transcribe_audio(audio_path)
            
            # Save the transcription (you can also compare it to the original transcription if available)
            transcriptions[audio_file] = transcription
            
            # Save transcription to a file
            transcription_file = os.path.join(transcription_dir, f"{audio_file.split('.')[0]}.txt")
            with open(transcription_file, "w") as f:
                f.write(transcription)
    
    return transcriptions

# Run the transcription for the entire dataset
if __name__ == "__main__":
    transcriptions = transcribe_dataset()
    print(f"Transcriptions have been saved to: {transcription_dir}")
