from fastapi import FastAPI
from pydantic import BaseModel

# Create the API
app = FastAPI(title="Cooking Tracker Voice Assistant")


# Define the incoming request format
class VoiceRequest(BaseModel):
    text: str


# Test whether the server is running
@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Cooking Tracker Voice Server Ready"
    }


# Receive a question and return a response
@app.post("/api/voice/query")
def voice_query(request: VoiceRequest):

    question = request.text.lower()

    print(f"Received question: {request.text}")

    # Temporary test responses

    if "cooking" in question:
        answer = "You are currently cooking eggs."

    elif "inventory" in question:
        answer = "Your inventory contains eggs, milk, and vegetables."

    elif "burn" in question:
        answer = (
            "I can monitor visual cooking activity, "
            "but I cannot reliably determine whether food is burning."
        )

    elif "hello" in question:
        answer = "Hello! I'm your cooking assistant."

    else:
        answer = "I'm ready to help you track your cooking."

    print(f"Response: {answer}")

    return {
        "transcript": request.text,
        "response": answer
    }
from fastapi import UploadFile, File
from pathlib import Path

@app.post("/api/voice/audio")
async def receive_audio(file: UploadFile = File(...)):

    audio_data = await file.read()

    output_path = Path("received_audio.wav")
    output_path.write_bytes(audio_data)

    print(f"Received audio: {len(audio_data)} bytes")

    return {
        "status": "success",
        "message": "Audio received successfully",
        "filename": str(output_path),
        "bytes": len(audio_data)
    }