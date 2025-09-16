import numpy as np
import sounddevice as sd
import wave
import struct

def generate_warning_sound(filename, duration=0.5, sample_rate=44100, frequency=880):
    """Generate a simple warning sound and save it as a WAV file."""
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate tone (sine wave)
    tone = 0.5 * np.sin(2 * np.pi * frequency * t)
    
    # Add a short silence
    silence = np.zeros(int(0.1 * sample_rate))
    
    # Repeat the tone with silence in between
    audio = np.concatenate([tone, silence, tone, silence, tone])
    
    # Convert to 16-bit integers
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Save as WAV file
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)   # 2 bytes per sample (16-bit)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

if __name__ == "__main__":
    generate_warning_sound("warning.wav")
