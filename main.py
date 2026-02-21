from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import pdfplumber
import os
from groq import Groq
from dotenv import load_dotenv

# -------------------- Load Environment Variables --------------------
load_dotenv()

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (fine for local dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=GROQ_API_KEY)

# Uploads directory
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- HELPERS --------------------
def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text.strip()


def generate_summary(text: str) -> str:
    """Generate a medical summary from the extracted lab report text."""
    prompt = f"""
You are a medical expert AI. Summarize this lab report for a patient in simple, clear language.
Include:
1. Key test results (with values and reference ranges)
2. Meaning or interpretation of each result
3. Overall health summary and recommendations.

Lab Report:
{text}
"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # ✅ Updated latest model
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print("Summary Error:", e)
        return f"Summary generation failed: {e}"


def translate_summary(text: str, target_lang: str) -> str:
    """Translate text into the target language using Groq LLM."""
    if target_lang == "en":
        return text  # No translation needed

    translation_prompt = f"Translate the following medical lab report summary into {target_lang}. Make it sound natural and medically accurate:\n\n{text}"
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # ✅ Updated latest model
            messages=[
                {"role": "system", "content": "You are a professional medical translator."},
                {"role": "user", "content": translation_prompt},
            ],
            temperature=0.3,
        )
        translated_text = response.choices[0].message.content.strip()
        return translated_text
    except Exception as e:
        print("Translation Error:", e)
        return f"Translation failed: {e}"


# -------------------- API ROUTES --------------------
import uuid

@app.post("/upload-report")
async def upload_report(file: UploadFile = File(...), language: str = Form("en")):
    try:
        # Unique file name
        unique_name = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_name)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        pdf_text = extract_text_from_pdf(file_path)

        if not pdf_text.strip():
            return {"summary": "No readable text found in the PDF file."}

        english_summary = generate_summary(pdf_text)
        final_summary = translate_summary(english_summary, language)

        # Clean storage
        os.remove(file_path)

        return {"summary": final_summary}

    except Exception as e:
        print("Error:", e)
        return {"summary": f"An error occurred: {str(e)}"}

# -------------------- ROOT TEST ROUTE --------------------
@app.get("/")
def root():
    return {"message": "✅ Multilingual Lab Report Summarizer API running successfully with LLaMA 3.3!"}
import uvicorn
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)