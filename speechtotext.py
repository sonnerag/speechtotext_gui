import pyaudio
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from threading import Thread
from google.oauth2 import service_account
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import translate_v2 as translate
from google.cloud import language_v2
import datetime
import document
from underthesea import word_tokenize

# Parameters for recording
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
CHUNK = 1024

# Global variables
is_recording = False
frames = []
full_transcription = ""  # Store all final transcribed text
interim_transcription = ""  # Store interim results temporarily

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Google Cloud credentials
client_file = 'SA_speech_demo.json'
credentials = service_account.Credentials.from_service_account_file(client_file)
speech_client = speech.SpeechClient(credentials=credentials)
translate_client = translate.Client(credentials=credentials)
language_client = language_v2.LanguageServiceClient(credentials=credentials)

# Streaming configuration
streaming_config = speech.StreamingRecognitionConfig(
    config=speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code='vi-VN',
        enable_automatic_punctuation=True,
        enable_speaker_diarization=True,
        diarization_speaker_count=2,
    ),
    interim_results=True,
)

# Function to translate text
def translate_text(text, target_language="en"):
    try:
        result = translate_client.translate(text, target_language=target_language)
        return result["translatedText"]
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# Function to summarize text using a simple heuristic (replace with GPT or NLP API)
def summarize_text(text):
    sentences = text.split(". ")
    if len(sentences) > 3:
        return ". ".join(sentences[:3]) + "..."
    return text

# Function to extract keywords using Google Cloud Natural Language API
def extract_keywords(text):
    document = language_v2.Document(
        content=text,
        type_=language_v2.Document.Type.PLAIN_TEXT,
    )
    response = language_client.analyze_entities(
        request={"document": document}
    )
    keywords = [entity.name for entity in response.entities if entity.type_.name in ["ORGANIZATION", "PERSON", "LOCATION", "EVENT", "WORK_OF_ART"]]
    return keywords

# Function to detect action items
def detect_action_items(text):
    action_phrases = ["cần phải", "làm ơn", "action item", "hãy", "phải"]
    action_items = [sentence for sentence in text.split(". ") if any(phrase in sentence for phrase in action_phrases)]
    return action_items

# Function to analyze sentiment
def analyze_sentiment(text):
    document = language_v2.Document(
        content=text,
        type_=language_v2.Document.Type.PLAIN_TEXT,
    )
    response = language_client.analyze_sentiment(
        request={"document": document}
    )
    return response.document_sentiment.score, response.document_sentiment.magnitude

# Function to handle streaming transcription
def stream_transcribe():
    global is_recording, full_transcription, interim_transcription
    is_recording = True

    # Open a streaming connection to Google Speech-to-Text
    requests = (
        speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in generate_audio_chunks()
    )
    responses = speech_client.streaming_recognize(streaming_config, requests)

    # Process responses
    for response in responses:
        if not response.results:
            continue

        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        if result.is_final:
            # Append final transcript to the full transcription with speaker tags and timestamps
            speaker_tag = get_speaker_tag(result)
            timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
            full_transcription += f"{timestamp} {speaker_tag}: {transcript}\n"
            interim_transcription = ""  # Clear interim result

            # Update additional features
            update_additional_features()
        else:
            # Update interim transcription
            interim_transcription = transcript

        # Update the transcription text area in the GUI
        transcription_text.delete(1.0, tk.END)
        transcription_text.insert(tk.END, full_transcription + interim_transcription)

    print("Streaming stopped.")

# Function to transcribe an uploaded audio file
def transcribe_audio_file(file_path):
    global full_transcription
    try:
        with open(file_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=RATE,
            language_code="vi-VN",
            enable_automatic_punctuation=True,
            enable_speaker_diarization=True,
            diarization_speaker_count=2,
        )

        response = speech_client.recognize(config=config, audio=audio)

        full_transcription = ""
        for result in response.results:
            speaker_tag = get_speaker_tag(result)
            transcript = result.alternatives[0].transcript
            full_transcription += f"{speaker_tag}: {transcript}\n"

        # Update the transcription text area in the GUI
        transcription_text.delete(1.0, tk.END)
        transcription_text.insert(tk.END, full_transcription)

        # Update additional features
        update_additional_features()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to transcribe audio file: {e}")

# Function to handle file upload
def upload_audio_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Audio Files", "*.wav *.mp3"), ("All Files", "*.*")]
    )
    if file_path:
        Thread(target=transcribe_audio_file, args=(file_path,)).start()

    # Function to extract speaker tag from the result
def get_speaker_tag(result):
    if not result.alternatives[0].words:
        return "Speaker ?"
    
    # Count the occurrences of each speaker tag in the result
    speaker_counts = {}
    for word in result.alternatives[0].words:
        speaker_tag = word.speaker_tag
        if speaker_tag in speaker_counts:
            speaker_counts[speaker_tag] += 1
        else:
            speaker_counts[speaker_tag] = 1
    
    # Determine the dominant speaker for this result
    dominant_speaker = max(speaker_counts, key=speaker_counts.get)
    return f"Speaker {dominant_speaker}"

# Function to generate audio chunks for streaming
def generate_audio_chunks():
    global is_recording
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    while is_recording:
        data = stream.read(CHUNK, exception_on_overflow=False)
        yield data

    stream.stop_stream()
    stream.close()

# Function to start recording and streaming
def start_recording():
    global full_transcription, interim_transcription
    full_transcription = ""  # Reset transcription when starting a new recording
    interim_transcription = ""  # Reset interim transcription
    # transcription_var.set("")  # Clear the label
    # topics_label.config(text="Detected Topics: ")  # Clear topics label
    # status_var.set("Recording...")  # Update status
    Thread(target=stream_transcribe).start()

# Function to stop recording
def stop_recording():
    global is_recording
    is_recording = False
    # Ensure the label displays the final transcription
    # transcription_var.set(full_transcription.strip())
    # status_var.set("Recording stopped.")  # Update status

# Function to update additional features (translation, summarization, etc.)
def update_additional_features():
    # Translate the transcription
    translated_text = translate_text(full_transcription, "en")
    translation_text.delete(1.0, tk.END)
    translation_text.insert(tk.END, translated_text)

    # Summarize the transcription
    summary = summarize_text(full_transcription)
    summary_text.delete(1.0, tk.END)
    summary_text.insert(tk.END, summary)

    # Extract keywords
    keywords = extract_keywords(full_transcription)
    keywords_text.delete(1.0, tk.END)
    keywords_text.insert(tk.END, ", ".join(keywords))

    # Detect action items
    action_items = detect_action_items(full_transcription)
    action_items_text.delete(1.0, tk.END)
    action_items_text.insert(tk.END, "\n".join(action_items))

    # Analyze sentiment
    sentiment_score, sentiment_magnitude = analyze_sentiment(full_transcription)
    sentiment_label.config(text=f"Sentiment: Score={sentiment_score}, Magnitude={sentiment_magnitude}")

# Function to export transcription to a file
def export_transcription():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt"), ("Word Documents", "*.docx")])
    if file_path:
        if file_path.endswith(".txt"):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(full_transcription)
        elif file_path.endswith(".docx"):
            doc = document()
            doc.add_paragraph(full_transcription)
            doc.save(file_path)
        print(f"Transcription exported to {file_path}")

# Tkinter GUI
root = tk.Tk()
root.title("Vietnamese Meeting Transcriber")
root.geometry("1000x800")

# Create a notebook (tabbed interface)
notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

# Tab 1: Transcription
transcription_tab = ttk.Frame(notebook)
notebook.add(transcription_tab, text="Transcription")

# Transcription text area
transcription_label = ttk.Label(transcription_tab, text="Transcription:")
transcription_label.pack(pady=5)
transcription_text = scrolledtext.ScrolledText(transcription_tab, wrap=tk.WORD, width=100, height=20)
transcription_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

# Tab 2: Translation
translation_tab = ttk.Frame(notebook)
notebook.add(translation_tab, text="Translation")

# Translation text area
translation_label = ttk.Label(translation_tab, text="Translation:")
translation_label.pack(pady=5)
translation_text = scrolledtext.ScrolledText(translation_tab, wrap=tk.WORD, width=100, height=20)
translation_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

# Tab 3: Summary
summary_tab = ttk.Frame(notebook)
notebook.add(summary_tab, text="Summary")

# Summary text area
summary_label = ttk.Label(summary_tab, text="Summary:")
summary_label.pack(pady=5)
summary_text = scrolledtext.ScrolledText(summary_tab, wrap=tk.WORD, width=100, height=10)
summary_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

# Tab 4: Keywords
keywords_tab = ttk.Frame(notebook)
notebook.add(keywords_tab, text="Keywords")

# Keywords text area
keywords_label = ttk.Label(keywords_tab, text="Keywords:")
keywords_label.pack(pady=5)
keywords_text = scrolledtext.ScrolledText(keywords_tab, wrap=tk.WORD, width=100, height=10)
keywords_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

# Tab 5: Action Items
action_items_tab = ttk.Frame(notebook)
notebook.add(action_items_tab, text="Action Items")

# Action items text area
action_items_label = ttk.Label(action_items_tab, text="Action Items:")
action_items_label.pack(pady=5)
action_items_text = scrolledtext.ScrolledText(action_items_tab, wrap=tk.WORD, width=100, height=10)
action_items_text.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

# Tab 6: Sentiment Analysis
sentiment_tab = ttk.Frame(notebook)
notebook.add(sentiment_tab, text="Sentiment Analysis")

# Sentiment label
sentiment_label = ttk.Label(sentiment_tab, text="Sentiment:")
sentiment_label.pack(pady=10)

# Buttons frame
buttons_frame = ttk.Frame(root)
buttons_frame.pack(pady=10)

# Start recording button
record_button = ttk.Button(buttons_frame, text="Start Recording", command=lambda: Thread(target=stream_transcribe).start())
record_button.grid(row=0, column=0, padx=5)

stop_button = ttk.Button(buttons_frame, text="Stop Recording", command=stop_recording)
stop_button.grid(row=0, column=1, padx=5)

# Upload audio file button
upload_button = ttk.Button(buttons_frame, text="Upload Audio File", command=upload_audio_file)
upload_button.grid(row=0, column=2, padx=5)

# Export transcription button
export_button = ttk.Button(buttons_frame, text="Export Transcription", command=export_transcription)
export_button.grid(row=0, column=3, padx=5)

# Exit button
exit_button = ttk.Button(buttons_frame, text="Exit", command=root.destroy)
exit_button.grid(row=0, column=4, padx=5)

root.mainloop()