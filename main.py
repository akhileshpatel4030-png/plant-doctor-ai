import os
import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types


app = FastAPI(title="Plant Doctor AI")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- HOME ----------------

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Plant Doctor AI API is running 🌱"
    }


# ---------------- PLANT DISEASE PREDICTION ----------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="कृपया plant की image upload करें।"
        )

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY Render में सेट नहीं है।"
        )

    try:
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Image खाली है।"
            )

        client = genai.Client(api_key=api_key)

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=file.content_type
        )

        prompt = """
You are Plant Doctor AI, an agricultural plant-disease assistant.

Analyze the uploaded plant/leaf image.

Return ONLY valid JSON in this exact format:

{
  "crop": "crop name",
  "disease": "disease name or Healthy",
  "confidence": 0,
  "description": "short Hindi description",
  "prevention": [
    "prevention 1",
    "prevention 2",
    "prevention 3"
  ],
  "next_steps": [
    "step 1",
    "step 2",
    "step 3"
  ]
}

Important:
- confidence must be a number from 0 to 100.
- If the image is unclear, use disease = "पहचान स्पष्ट नहीं है".
- Do not invent a disease when there is not enough evidence.
- Give the explanation in simple Hindi.
- Do not recommend unsafe or hazardous chemical use.
"""

        response = client.models.generate_content(
            model="gemini-3.7-flash",
            contents=[
                image_part,
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        text = response.text

        if not text:
            raise Exception("Gemini ने कोई response नहीं दिया।")

        result = json.loads(text)

        return {
            "success": True,
            "crop": result.get("crop", "Unknown"),
            "disease": result.get(
                "disease",
                "पहचान स्पष्ट नहीं है"
            ),
            "confidence": result.get("confidence", 0),
            "description": result.get(
                "description",
                "Image का analysis पूरा नहीं हो पाया।"
            ),
            "prevention": result.get("prevention", []),
            "next_steps": result.get("next_steps", [])
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="AI response सही format में नहीं मिला।"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI analysis error: {str(e)}"
        )


# ---------------- AI QUESTION FEATURE ----------------

class QuestionRequest(BaseModel):
    question: str
    crop: str = ""


@app.post("/ask")
async def ask_ai(data: QuestionRequest):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY Render में सेट नहीं है।"
        )

    question = data.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="कृपया अपना सवाल लिखें।"
        )

    if len(question) > 1000:
        raise HTTPException(
            status_code=400,
            detail="सवाल बहुत लंबा है। कृपया छोटा सवाल पूछें।"
        )

    crop = data.crop.strip()

    try:

        client = genai.Client(api_key=api_key)

        prompt = f"""
You are Plant Doctor AI.

You are an agricultural assistant for farmers.

Answer the user's question in simple Hindi.

Selected crop:
{crop if crop else "कोई crop select नहीं किया गया"}

User question:
{question}

Rules:
- Give practical and easy-to-understand agricultural information.
- If the question is about plant disease, explain possible causes and prevention.
- Do not claim that an AI answer is a guaranteed diagnosis.
- Do not provide unsafe or hazardous chemical instructions.
- If pesticide/fertilizer use is discussed, advise checking the product label
  and confirming with a qualified local agriculture expert.
- If the question cannot be answered reliably, say so clearly.
- Keep the answer concise but useful.
"""

        response = client.models.generate_content(
            model="gemini-3.7-flash",
            contents=prompt
        )

        answer = response.text

        if not answer:
            raise Exception("Gemini ने कोई answer नहीं दिया।")

        return {
            "success": True,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI question error: {str(e)}"
    )
