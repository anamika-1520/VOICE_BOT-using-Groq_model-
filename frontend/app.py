import streamlit as st
import numpy as np
import av
import tempfile
import requests
import os
import wave
import time
import scipy.signal as signal
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from groq import Groq
from gtts import gTTS
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Groq Voice Chat")
st.title("🎙️ Live Voice Chat (i am a voice bot and i am here to  answer your  questions)")

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# ---------------- AUDIO PROCESSOR ----------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []
        self.recording = False

    def recv(self, frame: av.AudioFrame):
        if self.recording:
            audio = frame.to_ndarray()
            self.frames.append(audio)
        return frame

ctx = webrtc_streamer(
    key="voice-chat",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

# ---------------- UI CONTROLS ----------------
if ctx.audio_processor:

    if st.button("🎙️ Start Speaking (5 sec)"):
        ctx.audio_processor.frames = []
        ctx.audio_processor.recording = True
        st.info("Speak clearly in English...")
        time.sleep(5)
        ctx.audio_processor.recording = False
        st.success("Recording done")

    if st.button("🧠 Ask"):
        if len(ctx.audio_processor.frames) == 0:
            st.error("No audio recorded")
        else:
            # Merge frames
            audio_np = np.concatenate(ctx.audio_processor.frames, axis=1)

            # Stereo → Mono
            if audio_np.ndim == 2:
                audio_np = audio_np.mean(axis=0)

            # Normalize
            audio_np = audio_np / np.max(np.abs(audio_np))

            # Resample 48k → 16k
            audio_16k = signal.resample_poly(audio_np, 16000, 48000)
            audio_16k = (audio_16k * 32767).astype(np.int16)

            # Write proper WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name

            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_16k.tobytes())

            # -------- GROQ SPEECH TO TEXT --------
            with open(wav_path, "rb") as audio_file:
                transcript = groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3",
                    language="en",
                    temperature=0.0
                )

            user_text = transcript.text
            st.success(f"You said: {user_text}")

            # -------- BACKEND CALL --------
            response = requests.post(
                "http://127.0.0.1:8000/ask",
                json={"question": user_text}
            )

            answer = response.json()["answer"]
            st.markdown("### 🤖 Anamika Says")
            st.write(answer)

            # -------- TEXT TO SPEECH --------
            tts = gTTS(answer, lang="en")
            tts.save("reply.mp3")
            st.audio("reply.mp3")
### this is the end of the code ###
### i             