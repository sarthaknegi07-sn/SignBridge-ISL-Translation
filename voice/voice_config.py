# Voice Assistant Configuration

# Voice options:
# Female Indian English: en-IN-NeerjaNeural
# Male Indian English:   en-IN-PrabhatNeural

VOICE = "en-IN-NeerjaNeural"
RATE = "+0%"          # speech speed: +10% faster, -10% slower
VOLUME = "+0%"        # volume: +10% louder, -10% quieter
PITCH = "+0Hz"        # pitch adjustment

# Detection settings
CONFIDENCE_THRESHOLD = 0.90   # only speak if model is 90%+ confident
COOLDOWN_SECONDS = 2.0        # wait 2s before speaking same phrase again

# Audio cache directory
AUDIO_CACHE_DIR = "voice/audio_cache"