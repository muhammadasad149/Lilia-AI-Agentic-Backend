# models.py
from sqlalchemy import Column, Integer, String, ForeignKey, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Doctor(Base):
    __tablename__ = "doctors"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    email = Column(String, nullable=True)
    patients = relationship("Patient", back_populates="doctor")

class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    age = Column(Integer)
    identity_number = Column(String)
    sex = Column(String)
    history = Column(String, nullable=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"))
    doctor = relationship("Doctor", back_populates="patients")
    scans = relationship("Scan", back_populates="patient", cascade="all, delete")

class Scan(Base):
    __tablename__ = "scans"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    upload_time = Column(String, default=lambda: datetime.utcnow().isoformat())
    patient_id = Column(Integer, ForeignKey("patients.id"))
    patient = relationship("Patient", back_populates="scans")
    result = Column(String, nullable=True)

class PredictionsDemographics(Base):
    __tablename__ = "predictions_demographics"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer)
    prediction = Column(String)
    confidence = Column(Float)
    actual_outcome = Column(String, nullable=True)
    age = Column(Integer)
    gender = Column(String)
    ethnicity = Column(String)
    scanner_model = Column(String)
    hospital_id = Column(String)
    timestamp = Column(String, default=lambda: datetime.utcnow().isoformat())

class BiasMetrics(Base):
    __tablename__ = "bias_metrics"
    id = Column(Integer, primary_key=True, index=True)
    analysis_date = Column(String)
    demographic_group = Column(String)
    group_type = Column(String)
    total_predictions = Column(Integer)
    accuracy = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1_score = Column(Float)
    false_positive_rate = Column(Float)
    false_negative_rate = Column(Float)

class BiasAlerts(Base):
    __tablename__ = "bias_alerts"
    id = Column(Integer, primary_key=True, index=True)
    alert_type = Column(String)
    severity = Column(String)
    message = Column(String)
    affected_groups = Column(String)
    timestamp = Column(String, default=lambda: datetime.utcnow().isoformat())
    resolved = Column(Boolean, default=False)
