from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine
from passlib.context import CryptContext
from jose import JWTError, jwt
from typing import Optional, List
from datetime import datetime, timedelta
import shutil
import os
from dotenv import load_dotenv
load_dotenv()

# --- Import models ---
from models import (
    Base, Doctor, Patient, Scan,
    PredictionsDemographics, BiasMetrics, BiasAlerts
)

# --- Import agents ---
from agents.omni_agent import full_prediction_pipeline
from agents.bias_detection_agent import BiasDetectionAgent

# --- Settings ---
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 12))
DATABASE_URL = os.getenv("DATABASE_URL")
UPLOAD_DIR = os.getenv("UPLOAD_DIR")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Database Setup ---
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)

# --- Security Utils ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_doctor(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    doctor = db.query(Doctor).filter(Doctor.username == username).first()
    if not doctor:
        raise credentials_exception
    return doctor

# --- FastAPI App ---
app = FastAPI(title="Lilia-AI Agentic Backend")

# --- Auth Endpoints ---
@app.post("/signup")
def signup(username: str = Form(...), password: str = Form(...), email: str = Form(None), db: Session = Depends(get_db)):
    if db.query(Doctor).filter(Doctor.username == username).first():
        raise HTTPException(400, "Username already exists")
    hashed = get_password_hash(password)
    doctor = Doctor(username=username, hashed_password=hashed, email=email)
    db.add(doctor)
    db.commit()
    db.refresh(doctor)
    return {"msg": "Signup successful", "doctor_id": doctor.id}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    doctor = db.query(Doctor).filter(Doctor.username == form_data.username).first()
    if not doctor or not verify_password(form_data.password, doctor.hashed_password):
        raise HTTPException(401, "Invalid credentials")
    access_token = create_access_token(data={"sub": doctor.username})
    return {"access_token": access_token, "token_type": "bearer"}

# --- Patient CRUD ---
@app.post("/patients")
def add_patient(
    name: str = Form(...),
    age: int = Form(...),
    identity_number: str = Form(...),
    sex: str = Form(...),
    history: str = Form(None),
    doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)):
    patient = Patient(name=name, age=age, identity_number=identity_number, sex=sex, history=history, doctor_id=doctor.id)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient

@app.get("/patients", response_model=List[dict])
def get_patients(doctor: Doctor = Depends(get_current_doctor), db: Session = Depends(get_db)):
    patients = db.query(Patient).filter(Patient.doctor_id == doctor.id).all()
    return [{"id": p.id, "name": p.name, "age": p.age, "identity_number": p.identity_number, "sex": p.sex, "history": p.history} for p in patients]

@app.get("/patients/{patient_id}")
def get_patient(patient_id: int, doctor: Doctor = Depends(get_current_doctor), db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == doctor.id).first()
    if not patient:
        raise HTTPException(404, "Patient not found")
    return patient

@app.put("/patients/{patient_id}")
def update_patient(
    patient_id: int,
    name: str = Form(None),
    age: int = Form(None),
    identity_number: str = Form(None),
    sex: str = Form(None),
    history: str = Form(None),
    doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == doctor.id).first()
    if not patient:
        raise HTTPException(404, "Patient not found")
    if name: patient.name = name
    if age: patient.age = age
    if identity_number: patient.identity_number = identity_number
    if sex: patient.sex = sex
    if history: patient.history = history
    db.commit()
    return patient

@app.delete("/patients/{patient_id}")
def delete_patient(patient_id: int, doctor: Doctor = Depends(get_current_doctor), db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == doctor.id).first()
    if not patient:
        raise HTTPException(404, "Patient not found")
    db.delete(patient)
    db.commit()
    return {"msg": "Deleted"}

# --- Scan Upload & Processing ---
@app.post("/patients/{patient_id}/upload_scan")
def upload_scan(
    patient_id: int,
    file: UploadFile = File(...),
    doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == doctor.id).first()
    if not patient:
        raise HTTPException(404, "Patient not found")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{UPLOAD_DIR}/{timestamp}_{file.filename}"

    with open(filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ai_result_raw = full_prediction_pipeline(filename, patient.id, db)
    ai_result = jsonable_encoder(ai_result_raw)  # 🔥 This line fixes the response
    scan = Scan(filename=filename, patient_id=patient.id, result=str(ai_result))
    db.add(scan)
    db.commit()
    db.refresh(scan)
    return {
        "msg": "Scan uploaded & processed",
        "scan_id": scan.id,
        "ai_result": ai_result
    }

@app.get("/patients/{patient_id}/scans")
def get_patient_scans(patient_id: int, doctor: Doctor = Depends(get_current_doctor), db: Session = Depends(get_db)):
    patient = db.query(Patient).filter(Patient.id == patient_id, Patient.doctor_id == doctor.id).first()
    if not patient:
        raise HTTPException(404, "Patient not found")
    return [{"scan_id": s.id, "filename": s.filename, "result": s.result} for s in patient.scans]

# --- Bias Feedback ---
class FeedbackRequest(BaseModel):
    actual_outcome: str

@app.post("/scans/{scan_id}/feedback")
def provide_feedback(
    scan_id: int,
    feedback: FeedbackRequest,
    doctor: Doctor = Depends(get_current_doctor),
    db: Session = Depends(get_db)):
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    patient = db.query(Patient).filter(Patient.id == scan.patient_id, Patient.doctor_id == doctor.id).first()
    if not patient:
        raise HTTPException(status_code=403, detail="Not authorized to provide feedback for this scan")

    try:
        bias_agent = BiasDetectionAgent()
        bias_agent.update_actual_outcome(db=db, patient_id=scan.patient_id, actual_outcome=feedback.actual_outcome)
        return {"msg": "Feedback received and logged successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to log feedback: {str(e)}")

# --- Root Endpoint ---
@app.get("/")
def root():
    return {"msg": "Lilia-AI Backend Running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
