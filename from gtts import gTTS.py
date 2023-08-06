from gtts import gTTS
import pygame
from io import BytesIO

def text_to_speech(text, language='en'):
    tts = gTTS(text=text, lang=language, slow=False)
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)  

    # Initialize the pygame mixer
    pygame.mixer.init() 
    pygame.mixer.music.load(audio_bytes)
    pygame.mixer.music.play()

    # Wait for the audio to finish playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

text_to_read = "Hello, this is a sample text. I am converting text to speech using Python."
text_to_speech(text_to_read)
