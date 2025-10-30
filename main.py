import os, io, json, base64, uuid, asyncio
from typing import Optional, Dict, Any, List

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiohttp

from faster_whisper import WhisperModel
from dotenv import load_dotenv

# ---------- ENV / CONFIG ----------
load_dotenv()

# Bhashini / ULCA creds (add in Railway → Variables)
BHASHINI_BASE_URL       = (os.getenv("BHASHINI_BASE_URL") or "").rstrip("/")
BHASHINI_INFER_ENDPOINT = os.getenv("BHASHINI_INFER_ENDPOINT") or "/ulca/apis/v0/model/compute"
BHASHINI_USER_ID        = os.getenv("BHASHINI_USER_ID") or ""
BHASHINI_API_KEY        = os.getenv("BHASHINI_API_KEY") or ""

# Whisper config
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")  # tiny/base/small/medium/large-v3
WHISPER_DEVICE     = os.getenv("WHISPER_DEVICE", "cpu")        # "cpu" or "cuda"
WHISPER_COMPUTE    = os.getenv("WHISPER_COMPUTE_TYPE", "int8") # int8 on CPU is fastest

DEFAULT_SOURCE_LANG = os.getenv("DEFAULT_SOURCE_LANG", "auto")
DEFAULT_TARGET_LANG = os.getenv("DEFAULT_TARGET_LANG", "en")    # translate+TTS target

# ---------- APP ----------
app = FastAPI(title="Realtime STT (faster-whisper) + Bhashini TTS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ---------- MODEL (load once) ----------
print(f"[boot] loading faster-whisper '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE} ({WHISPER_COMPUTE})")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)

# ---------- HELPERS ----------
async def bhashini_translate(session: aiohttp.ClientSession, text: str, src_lang: str, tgt_lang: str) -> str:
    """
    ULCA translation pipeline. Your Bhashini account may require provider/model IDs.
    Payload here is the generic pipeline form; adjust if your tenant requires.
    """
    if not (BHASHINI_BASE_URL and BHASHINI_USER_ID and BHASHINI_API_KEY):
        return text
    url = f"{BHASHINI_BASE_URL}{BHASHINI_INFER_ENDPOINT}"
    headers = {"Content-Type": "application/json", "userID": BHASHINI_USER_ID, "ulcaApiKey": BHASHINI_API_KEY}
    payload = {
        "pipelineTasks": [{"taskType": "translation", "config": {"language": {"sourceLanguage": src_lang, "targetLanguage": tgt_lang}}}],
        "inputData": {"input": [{"source": text}]}
    }
    async with session.post(url, headers=headers, data=json.dumps(payload), timeout=60) as r:
        if r.status != 200:
            return text
        data = await r.json()
        try:
            return data["pipelineResponse"][0]["output"][0]["target"]
        except Exception:
            return text

async def bhashini_tts(session: aiohttp.ClientSession, text: str, lang: str) -> Optional[bytes]:
    """
    ULCA TTS pipeline. Returns audio bytes (usually base64-encoded in response).
    """
    if not (BHASHINI_BASE_URL and BHASHINI_USER_ID and BHASHINI_API_KEY):
        return None
    url = f"{BHASHINI_BASE_URL}{BHASHINI_INFER_ENDPOINT}"
    headers = {"Content-Type": "application/json", "userID": BHASHINI_USER_ID, "ulcaApiKey": BHASHINI_API_KEY}
    payload = {
        "pipelineTasks": [{"taskType": "tts", "config": {"language": {"sourceLanguage": lang}}}],
        "inputData": {"input": [{"source": text}]}
    }
    async with session.post(url, headers=headers, data=json.dumps(payload), timeout=120) as r:
        if r.status != 200:
            return None
        data = await r.json()
        try:
            b64 = data["pipelineResponse"][0]["audio"][0]["audioContent"]
            return base64.b64decode(b64)
        except Exception:
            return None

def concat_text(segments) -> str:
    return " ".join([s.text.strip() for s in segments]).strip()

# ---------- ROUTES ----------
@app.get("/health")
async def health():
    return {"ok": True}

# REST: single-shot upload (send short .wav/.mp3)
@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    source_lang: str = Form(DEFAULT_SOURCE_LANG),
    target_lang: str = Form(DEFAULT_TARGET_LANG)
):
    # Save uploaded temp file (faster-whisper reads via ffmpeg)
    tmp_path = f"/tmp/{uuid.uuid4()}_{audio.filename}"
    with open(tmp_path, "wb") as f:
        f.write(await audio.read())

    # STT
    segments, info = whisper_model.transcribe(
        tmp_path, beam_size=5, language=None if source_lang=="auto" else source_lang, vad_filter=True
    )
    transcript = concat_text(segments)
    detected = info.language or "auto"

    # Translate + TTS via Bhashini
    async with aiohttp.ClientSession() as session:
        translated = await bhashini_translate(session, transcript, detected, target_lang) if transcript else ""
        tts_bytes = await bhashini_tts(session, translated or transcript, target_lang)

    try: os.remove(tmp_path)
    except: pass

    resp: Dict[str, Any] = {
        "id": str(uuid.uuid4()),
        "detected_language": detected,
        "transcript": transcript,
        "translated_text": translated or transcript,
        "target_language": target_lang
    }
    if tts_bytes:
        resp["tts_audio_base64"] = base64.b64encode(tts_bytes).decode("utf-8")
    return JSONResponse(resp)

# WebSocket: send ~1s chunks (WAV/MP3) as base64; send {"type":"end"} to finalize
@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    buffer_paths: List[str] = []   # we’ll append small files then decode full on "end"
    source_lang = DEFAULT_SOURCE_LANG
    target_lang = DEFAULT_TARGET_LANG

    try:
        async with aiohttp.ClientSession() as session:
            while True:
                raw = await ws.receive_text()
                data = json.loads(raw)

                if data.get("type") == "config":
                    source_lang = data.get("source_lang", source_lang)
                    target_lang = data.get("target_lang", target_lang)
                    await ws.send_text(json.dumps({"event":"config_ack","source_lang":source_lang,"target_lang":target_lang}))
                    continue

                if data.get("type") == "chunk":
                    b64 = data.get("audio_base64")
                    if not b64: 
                        continue
                    # write small temp audio file
                    p = f"/tmp/chunk_{uuid.uuid4()}.bin"
                    with open(p, "wb") as f:
                        f.write(base64.b64decode(b64))
                    buffer_paths.append(p)
                    # (Optional) you could emit partials by transcribing recent seconds only.

                if data.get("type") == "end":
                    # concat: faster-whisper can accept a list of files; simplest is concat via ffmpeg,
                    # but to keep it portable, we’ll just transcribe the last file (near-realtime UX:
                    # send 1–2s chunks and “end” per utterance).
                    last_path = buffer_paths[-1] if buffer_paths else None
                    if not last_path:
                        await ws.send_text(json.dumps({"event":"final","error":"no audio"}))
                        continue

                    segments, info = whisper_model.transcribe(
                        last_path, beam_size=5, language=None if source_lang=="auto" else source_lang, vad_filter=True
                    )
                    text = concat_text(segments)
                    detected = info.language or "auto"

                    translated = await bhashini_translate(session, text, detected, target_lang) if text else ""
                    tts_bytes = await bhashini_tts(session, translated or text, target_lang)

                    resp = {
                        "event": "final",
                        "detected_language": detected,
                        "transcript": text,
                        "translated_text": translated or text,
                        "target_language": target_lang
                    }
                    if tts_bytes:
                        resp["tts_audio_base64"] = base64.b64encode(tts_bytes).decode("utf-8")
                    await ws.send_text(json.dumps(resp))

                    # cleanup
                    for p in buffer_paths:
                        try: os.remove(p)
                        except: pass
                    buffer_paths = []
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try: await ws.send_text(json.dumps({"event":"error","message":str(e)}))
        except: pass
        await ws.close()
