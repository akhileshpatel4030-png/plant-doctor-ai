from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Plant Doctor AI")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Plant Doctor AI API is running 🌱"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    return {
        "success": True,
        "crop": "Tomato",
        "disease": "Leaf Disease (Demo)",
        "confidence": 82,
        "description": "यह अभी demo result है। वास्तविक AI model बाद में जोड़ा जाएगा।",
        "prevention": [
            "पौधे की नियमित जाँच करें।",
            "बीमारी वाले पत्तों पर ध्यान दें।",
            "खेत में उचित सफाई और निगरानी रखें।",
            "जरूरत पड़ने पर कृषि विशेषज्ञ से सलाह लें।"
        ],
        "next_steps": [
            "अच्छी रोशनी में पौधे की साफ फोटो लें।",
            "लक्षणों की निगरानी करें।",
            "उपचार से पहले बीमारी की पुष्टि करें।"
        ]
    }
