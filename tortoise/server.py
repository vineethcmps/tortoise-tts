import os
import time

import torch
import torchaudio
import uuid

from api import TextToSpeech, MODELS_DIR
from utils.audio import load_voices
from utils.text import split_and_recombine_text
from flask import Flask, request, jsonify
import random
from io import BytesIO

tts = TextToSpeech(models_dir=MODELS_DIR)

app = Flask(__name__)

voice_to_gender = {
    "male": ["daniel", "deniro", "freeman", "geralt"],
    "female": ["angie", "emma", "halle", "jlaw"]
}

def do_tts(text, gender, preset, voice_sel=None):
    name= str(uuid.uuid4())
    if voice_sel is None:
        voice_sel = random.choice(voice_to_gender[gender])
    voice_samples, conditioning_latents = load_voices([voice_sel])
    gen, dbg_state = tts.tts_with_preset(text, k=1, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                preset=preset, use_deterministic_seed=None, return_deterministic_state=True, cvvp_amount=.0)
    # code to generate story based on the script
    out_file = f"/out/{uuid.uuid4()}.wav"
    torchaudio.save(out_file, gen.squeeze(0).cpu(), 24000, format="wav")
    return out_file, voice_sel


def do_tts_paragraph(text, gender, preset, voice_sel=None):
    if voice_sel is None:
        voice_sel = random.choice(voice_to_gender[gender])
    texts = split_and_recombine_text(text)
    voice_samples, conditioning_latents = load_voices([voice_sel])

    all_parts = []
    seed = int(time.time())
    for j, text in enumerate(texts):
        gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                  preset=preset, k=1, use_deterministic_seed=seed, temperature=0)
        gen = gen.squeeze(0).cpu()
        all_parts.append(gen)

    full_audio = torch.cat(all_parts, dim=-1)
    out_file = f"/out/{uuid.uuid4()}.wav"
    torchaudio.save(out_file, full_audio, 24000, format="wav")
    return out_file, voice_sel


@app.route("/get_voice", methods=["POST"])
def get_voice():
    text = request.json.get("text")
    gender = request.json.get("gender", "male")
    preset = request.json.get("preset", "fast")
    voice_sel = request.json.get("voice_sel", None)

    out_file, voice_sel = do_tts(text, gender, preset, voice_sel)
    return jsonify({"out_file": out_file, "voice_sel": voice_sel})


@app.route("/get_voice_paragraph", methods=["POST"])
def get_voice_paragraph():
    text = request.json.get("text")
    gender = request.json.get("gender", "male")
    preset = request.json.get("preset", "fast")
    voice_sel = request.json.get("voice_sel", None)

    out_file, voice_sel = do_tts_paragraph(text, gender, preset, voice_sel)
    return jsonify({"out_file": out_file, "voice_sel": voice_sel})


if __name__ == "__main__":
    # app.run(debug=True)
    do_tts("hello","male","fast")
    app.run(host="0.0.0.0")
