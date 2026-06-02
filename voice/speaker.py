import os
import asyncio
import edge_tts
import pygame
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voice.voice_config import VOICE, RATE, VOLUME, AUDIO_CACHE_DIR

os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

def get_cache_path(text):
    safe_name = text.replace(" ", "_").replace("/", "_")
    return os.path.join(AUDIO_CACHE_DIR, f"{safe_name}.mp3")

async def generate_audio(text, output_path):
    communicate = edge_tts.Communicate(text, voice=VOICE, rate=RATE, volume=VOLUME)
    await communicate.save(output_path)

def speak(text):
    try:
        cache_path = get_cache_path(text)

        # Generate if not cached
        if not os.path.exists(cache_path):
            asyncio.run(generate_audio(text, cache_path))

        # Play audio
        pygame.mixer.init()
        pygame.mixer.music.load(cache_path)
        pygame.mixer.music.play()

        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except Exception as e:
        print(f"Speaker error: {e}")

def precache_phrases(phrases):
    """Pre-generate audio for all phrases at startup"""
    print("Pre-caching audio for all phrases...")
    for phrase in phrases:
        cache_path = get_cache_path(phrase)
        if not os.path.exists(cache_path):
            asyncio.run(generate_audio(phrase, cache_path))
            print(f"  Cached: {phrase}")
    print("Audio cache ready!")