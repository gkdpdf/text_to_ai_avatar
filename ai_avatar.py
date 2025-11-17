# ai_avatar.py
import os
import sys
import argparse
from moviepy.editor import ImageClip, AudioFileClip

# Add Wav2Lip folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), "Wav2Lip"))

from inference import load_model, face_detect, datagen, _load_image

# ===============================
# 🔥 Run Wav2Lip function
# ===============================
def run_wav2lip(face_path, audio_path, checkpoint_path, out_path):
    # Load model
    model = load_model(checkpoint_path)
    print("✅ Model loaded")

    # Load image
    img = _load_image(face_path)

    # Load audio and convert to mel spectrogram
    import audio
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)

    # Generate frames (simplified)
    gen = datagen(img, mel)
    frames = list(gen)
    print(f"✅ Generated {len(frames)} frames")

    # Convert frames to video using moviepy
    clip = ImageClip(frames[0]).set_duration(AudioFileClip(audio_path).duration)
    clip = clip.set_audio(AudioFileClip(audio_path))
    clip.write_videofile(out_path, fps=25)
    print(f"🎉 Done! Saved output: {out_path}")

# ===============================
# 🔥 Main
# ===============================
def main():
    parser = argparse.ArgumentParser(description="AI Avatar with Wav2Lip")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to Wav2Lip.pth checkpoint")
    parser.add_argument("--face", type=str, required=True, help="Path to face image (jpg/png)")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (wav/mp3)")
    parser.add_argument("--out", type=str, default="output.mp4", help="Output video file")
    args = parser.parse_args()

    run_wav2lip(args.face, args.audio, args.checkpoint_path, args.out)

if __name__ == "__main__":
    main()
