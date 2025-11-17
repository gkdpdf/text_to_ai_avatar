from TTS.api import TTS
import torch

print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Hindi-capable XTTS model
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

tts = TTS(model_name).to("cuda")

text = '''गुरु,मैं सिद्धू जी की आवाज़ की क्लोन हूँ,और मैं क्रिकमीमप्ले के लिए काम करता हूँ।और मैं इनसे एक रुपये की दिहाड़ी भी नहीं लेता। ठोक्को ताली!

'''
speaker_wav = "sidhujii.wav"

tts.tts_to_file(
    text=text,
    speaker_wav=speaker_wav,
    file_path="output7.wav",
    language="hi"
)

print("DONE: output7.wav")
