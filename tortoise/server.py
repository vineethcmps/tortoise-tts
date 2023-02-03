import os

import torch
import torchaudio

from api import TextToSpeech, MODELS_DIR
from utils.audio import load_voices
from flask import Flask, request, make_response
import random
from io import BytesIO

tts = TextToSpeech(models_dir=MODELS_DIR)

app = Flask(__name__)

voice_to_gender = {
    "male": ["daniel", "deniro", "freeman", "geralt"],
    "female": ["angie", "emma", "halle", "jlaw"]
}

def do_tts(text, gender, preset):
    voice_sel = random.choice(voice_to_gender[gender])
    voice_samples, conditioning_latents = load_voices([voice_sel])
    gen, dbg_state = tts.tts_with_preset(text, k=1, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                preset=preset, use_deterministic_seed=None, return_deterministic_state=True, cvvp_amount=.0)
    # code to generate story based on the script

    byts = BytesIO()
    torchaudio.save(byts, gen.squeeze(0).cpu(), 24000, format="wav")
    return byts


@app.route("/get_voice", methods=["POST"])
def get_voice():
    text = request.json.get("text")
    gender = request.json.get("gender", "male")
    preset = request.json.get("preset", "fast")

    audio_bytes = do_tts(text, gender, preset)
    response = make_response(audio_bytes.getvalue())
    audio_bytes.close()
    response.headers['Content-Type'] = 'audio/wav'
    response.headers['Content-Disposition'] = 'attachment; filename=sound.wav'
    return response


if __name__ == "__main__":
    # app.run(debug=True)
    do_tts("hello","male","fast")
    app.run(host="0.0.0.0")
