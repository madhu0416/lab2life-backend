from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import pdfplumber
import os
import uuid
import json
import re
from groq import Groq
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta

# -------------------- Load Environment Variables --------------------
load_dotenv()

app = FastAPI()

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://lab2life-frontendd.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- GROQ CLIENT --------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=GROQ_API_KEY)

# -------------------- SECURITY --------------------
SECRET_KEY = os.getenv("SECRET_KEY", "lab2life-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 12

pwd_context = CryptContext(schemes=["bcrypt_sha256"], deprecated="auto")
security = HTTPBearer()

# -------------------- DATABASE --------------------
DATABASE_URL = "sqlite:///./lab2life.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# -------------------- UPLOADS DIRECTORY --------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- LANGUAGE MAP --------------------
LANGUAGE_MAP = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
    "bn": "Bengali",
    "gu": "Gujarati",
    "te": "Telugu",
    "fr": "French",
    "es": "Spanish",
    "ar": "Arabic",
}

# -------------------- DATABASE MODELS --------------------
class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(20), nullable=False)
    phone = Column(String(10), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)

    reports = relationship("Report", back_populates="patient")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(255), nullable=False)
    language = Column(String(20), nullable=False)
    summary = Column(Text, nullable=True)
    health_score = Column(Integer, nullable=True)
    risk_level = Column(String(100), nullable=True)
    normal_factors = Column(Text, nullable=True)
    abnormal_factors = Column(Text, nullable=True)
    recommendations = Column(Text, nullable=True)
    doctor_advice = Column(Text, nullable=True)

    patient = relationship("Patient", back_populates="reports")


Base.metadata.create_all(bind=engine)

# -------------------- REQUEST MODELS --------------------
class AskDoctorRequest(BaseModel):
    question: str
    summary: str


class RegisterRequest(BaseModel):
    full_name: str
    age: int
    gender: str
    phone: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# -------------------- DATABASE HELPERS --------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_patient(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    token = credentials.credentials

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        patient_id = payload.get("sub")
        if patient_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    patient = db.query(Patient).filter(Patient.id == int(patient_id)).first()
    if not patient:
        raise HTTPException(status_code=401, detail="Patient not found")

    return patient


# -------------------- REPORT HELPERS --------------------
def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    return text.strip()


def extract_json_from_response(content: str):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Model did not return valid JSON")


def generate_report_analysis(text: str, target_lang: str = "en") -> dict:
    language_name = LANGUAGE_MAP.get(target_lang, "English")

    prompt = f"""
You are a careful medical report assistant.

Analyze the following lab report and return ONLY valid JSON.
Do not return markdown.
Do not return explanation outside JSON.

The response language must be: {language_name}

Return this exact JSON structure:
{{
  "summary": "detailed patient-friendly explanation of the report",
  "health_score": 85,
  "risk_level": "Low Risk",
  "normal_factors": ["factor 1", "factor 2"],
  "abnormal_factors": ["factor 1", "factor 2"],
  "recommendations": ["suggestion 1", "suggestion 2", "suggestion 3"],
  "doctor_advice": "short doctor advice"
}}

Rules:
- Base everything only on the actual report text.
- Keep language simple and patient-friendly.
- The summary must be detailed, not just 1 line.
- The summary should be 5 to 10 sentences minimum.
- Mention actual values and reference ranges whenever possible from the report.
- Explain important test results, whether they are normal or abnormal, what they may indicate, and give an overall understanding.
- health_score must be an integer from 0 to 100.
- risk_level should be short, like Low Risk / Moderate Risk / High Risk.
- normal_factors must contain only findings that are normal.
- abnormal_factors must contain only findings that are abnormal or need attention.
- recommendations must be practical and relevant to the report.
- doctor_advice must be realistic and safe.
- If data is missing, be cautious and do not invent too much.
- All text values must be in {language_name}.

Lab Report:
{text}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful medical AI that returns only valid JSON."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

        content = response.choices[0].message.content.strip()
        parsed = extract_json_from_response(content)

        health_score = parsed.get("health_score", 0)
        try:
            health_score = int(health_score)
        except Exception:
            health_score = 0

        if health_score < 0:
            health_score = 0
        if health_score > 100:
            health_score = 100

        return {
            "summary": parsed.get("summary", ""),
            "health_score": health_score,
            "risk_level": parsed.get("risk_level", ""),
            "normal_factors": parsed.get("normal_factors", []),
            "abnormal_factors": parsed.get("abnormal_factors", []),
            "recommendations": parsed.get("recommendations", []),
            "doctor_advice": parsed.get("doctor_advice", ""),
        }

    except Exception as e:
        print("Report Analysis Error:", e)
        return {
            "summary": f"Analysis generation failed: {str(e)}",
            "health_score": 0,
            "risk_level": "Unknown",
            "normal_factors": [],
            "abnormal_factors": [],
            "recommendations": [],
            "doctor_advice": "Unable to generate doctor advice.",
        }


def generate_doctor_answer(question: str, summary: str) -> str:
    prompt = f"""
You are a helpful medical assistant.

Use the report summary below and answer the user's question in simple, patient-friendly language.

Report Summary:
{summary}

User Question:
{question}

Rules:
- Keep the answer simple and clear.
- Do not give a final diagnosis.
- Be safe and practical.
- Suggest consulting a doctor when needed.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Ask Doctor Error:", e)
        return f"Unable to generate answer: {str(e)}"


# -------------------- AUTH ROUTES --------------------
@app.post("/register")
def register(data: RegisterRequest, db: Session = Depends(get_db)):
    if not data.full_name.replace(" ", "").isalpha() or len(data.full_name.strip()) < 2:
        raise HTTPException(status_code=400, detail="Invalid full name")

    if data.age < 1 or data.age > 120:
        raise HTTPException(status_code=400, detail="Invalid age")

    if data.gender not in ["Male", "Female", "Other"]:
        raise HTTPException(status_code=400, detail="Invalid gender")

    if not data.phone.isdigit() or len(data.phone) != 10:
        raise HTTPException(status_code=400, detail="Phone number must be exactly 10 digits")

    if len(data.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    existing_email = db.query(Patient).filter(Patient.email == data.email).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")

    existing_phone = db.query(Patient).filter(Patient.phone == data.phone).first()
    if existing_phone:
        raise HTTPException(status_code=400, detail="Phone number already registered")

    patient = Patient(
        full_name=data.full_name.strip(),
        age=data.age,
        gender=data.gender,
        phone=data.phone,
        email=data.email,
        password_hash=hash_password(data.password),
    )

    db.add(patient)
    db.commit()
    db.refresh(patient)

    return {"message": "Patient registered successfully"}


@app.post("/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.email == data.email).first()

    if not patient or not verify_password(data.password, patient.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"sub": str(patient.id)})

    return {
        "access_token": token,
        "token_type": "bearer",
        "full_name": patient.full_name,
    }


# -------------------- PUBLIC UPLOAD ROUTE (NOT SAVED FOR ANONYMOUS USERS) --------------------
@app.post("/upload-report")
async def upload_report(
    file: UploadFile = File(...),
    language: str = Form("en"),
):
    file_path = None

    try:
        if not file.filename.lower().endswith(".pdf"):
            return {
                "summary": "Only PDF files are supported right now.",
                "health_score": 0,
                "risk_level": "Unknown",
                "normal_factors": [],
                "abnormal_factors": [],
                "recommendations": [],
                "doctor_advice": "Please upload a PDF lab report.",
            }

        unique_name = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_name)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        pdf_text = extract_text_from_pdf(file_path)

        if not pdf_text.strip():
            return {
                "summary": "No readable text found in the PDF file.",
                "health_score": 0,
                "risk_level": "Unknown",
                "normal_factors": [],
                "abnormal_factors": [],
                "recommendations": [],
                "doctor_advice": "Please upload a clearer PDF report.",
            }

        analysis = generate_report_analysis(pdf_text, language)
        return analysis

    except Exception as e:
        print("Upload Report Error:", e)
        return {
            "summary": f"An error occurred: {str(e)}",
            "health_score": 0,
            "risk_level": "Unknown",
            "normal_factors": [],
            "abnormal_factors": [],
            "recommendations": [],
            "doctor_advice": "Unable to generate doctor advice.",
        }

    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as cleanup_error:
                print("File cleanup error:", cleanup_error)


# -------------------- PROTECTED ASK DOCTOR ROUTE --------------------
@app.post("/ask-doctor")
async def ask_doctor(
    data: AskDoctorRequest,
    current_patient: Patient = Depends(get_current_patient),
):
    try:
        answer = generate_doctor_answer(data.question, data.summary)
        return {"answer": answer}
    except Exception as e:
        print("Ask Doctor Route Error:", e)
        return {"answer": f"An error occurred: {str(e)}"}


# -------------------- ROOT TEST ROUTE --------------------
@app.get("/")
def root():
    return {
        "message": "Lab2Life API is running successfully with public upload and protected Ask Doctor."
    }


# -------------------- RUN APP --------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)