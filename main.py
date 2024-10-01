import os
import pyaudio
import wave
import nltk
from google.cloud import speech
from tkinter import Tk, Label, Entry, Button, StringVar
import threading
from nltk.translate.bleu_score import SmoothingFunction

nltk.download('punkt')

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:\AI_Project\Speech2Text\logical-signer-404016-415212da26e1.json"

is_recording = False
frames = []

def start_recording():
    global is_recording, frames
    is_recording = True
    frames = []

    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

    while is_recording:
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

def stop_recording(filename="recording.wav"):
    global is_recording
    is_recording = False

    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
    wf.setframerate(44100)
    wf.writeframes(b''.join(frames))
    wf.close()



def transcribe_audio(filename="recording.wav"):
    client = speech.SpeechClient()

    # Đọc file âm thanh
    with open(filename, 'rb') as audio_file:
        content = audio_file.read()

    # Kiểm tra dữ liệu gửi đi
    print(f"Đã gửi {len(content)} byte dữ liệu.")

    audio = speech.RecognitionAudio(content=content)

    # Cấu hình yêu cầu
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="en-US"
    )


    try:
        response = client.recognize(config=config, audio=audio)

      
        if not response.results:
            print("Không có kết quả nhận dạng.")
            return ""

        
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript

        print("Transcript: {}".format(transcript))
        return transcript

    except Exception as e:
        print(f"Lỗi khi gọi Google Speech-to-Text API: {e}")
        return ""




# Tính điểm BLEU
def calculate_bleu(reference, hypothesis):
    reference_tokenized = nltk.word_tokenize(reference.lower())
    hypothesis_tokenized = nltk.word_tokenize(hypothesis.lower())

    
    chencherry = SmoothingFunction()

    # tính điểm BLEU
    return nltk.translate.bleu_score.sentence_bleu([reference_tokenized], hypothesis_tokenized, 
                                                   smoothing_function=chencherry.method7)

# Khởi tạo giao diện người dùng
root = Tk()
root.title("Đánh giá phát âm tiếng Anh")

# Thiết lập kích thước cửa sổ
root.geometry("600x600")

target_text = StringVar()
entry = Entry(root, textvariable=target_text)
entry.pack()

result_label = Label(root, text="")
result_label.pack()

record_button = Button(root, text="Record", command=lambda: threading.Thread(target=start_recording).start())
record_button.pack()

stop_button = Button(root, text="Stop and Evaluate", command=lambda: [stop_recording(), evaluate_pronunciation()])
stop_button.pack()

# Thêm phần còn lại của mã nguồn của bạn ở đây



# Function to evaluate pronunciation
def evaluate_pronunciation():
    # Stop recording and save the audio file
    stop_recording()

    # Transcribe the recorded audio
    transcribed_text = transcribe_audio("recording.wav")
    if not transcribed_text:
        result_label.config(text="Không thể nhận dạng văn bản từ âm thanh.")
        return
    # Get the target text entered by the user
    target = target_text.get()

    # Calculate BLEU score
    bleu_score = calculate_bleu(target, transcribed_text)

    # Display the result
    result_label.config(text=f"BLEU Score: {bleu_score:.2f}")
# Run the application
root.mainloop()
