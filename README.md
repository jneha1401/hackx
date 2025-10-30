# Realtime STT (faster-whisper) + Bhashini TTS for Railway

## Endpoints
- GET /health
- POST /transcribe  (form-data: audio=@file.wav, source_lang=auto, target_lang=en)
- WS  /ws
  - send: {"type":"config","source_lang":"hi","target_lang":"en"} (optional)
  - send: {"type":"chunk","audio_base64":"<base64 small 1-2s audio>"}
  - send: {"type":"end"}
  - receive: {"event":"final","transcript":"...","translated_text":"...","tts_audio_base64":"..."} 

## Run locally
docker build -t rt-bridge .
docker run -it -p 8000:8000 --env-file .env rt-bridge
# open http://localhost:8000/health

## Frontend snippet
const ws = new WebSocket("wss://YOUR-RAILWAY-URL/ws");
ws.onopen = () => ws.send(JSON.stringify({type:"config",source_lang:"hi",target_lang:"en"}));
function sendChunk(b64) { ws.send(JSON.stringify({type:"chunk",audio_base64:b64})); }
function stopUtterance() { ws.send(JSON.stringify({type:"end"})); }
