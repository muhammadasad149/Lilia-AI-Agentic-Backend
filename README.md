# Lilia-AI Agentic Backend

An advanced AI-powered medical imaging analysis system for automated detection, segmentation, and clinical decision support for Cerebral Arteriovenous Malformations (AVM) from CT scans.

## 🚀 Features

- **Multi-Agent AI Pipeline**: Orchestrated workflow with 13 specialized AI agents
- **Real-time CT Scan Analysis**: Automated aneurysm detection and segmentation
- **Clinical Decision Support**: Risk assessment, treatment recommendations, and prioritization
- **Bias Detection & Monitoring**: Continuous monitoring for algorithmic bias
- **Explainable AI**: Detailed explanations of AI predictions
- **Clinical Trial Matching**: Automatic matching to relevant clinical trials
- **Billing Optimization**: Automated CPT code generation
- **Secure API**: JWT-based authentication with role-based access
- **Docker Deployment**: Containerized for easy deployment

## 🏗️ Architecture

### Core Components

#### 1. FastAPI Backend (`main.py`)
- RESTful API with JWT authentication
- Patient and scan management
- Secure file upload handling
- Real-time AI processing pipeline

#### 2. Database Models (`models.py`)
- **Doctor**: User authentication and management
- **Patient**: Patient demographics and medical history
- **Scan**: CT scan metadata and AI results
- **PredictionsDemographics**: Bias tracking and demographics
- **BiasMetrics**: Performance metrics by demographic groups
- **BiasAlerts**: Automated bias detection alerts

#### 3. AI Agent System (`agents/`)

##### **Omni Agent** (`omni_agent.py`)
Main orchestrator that coordinates the entire 11-stage analysis pipeline:
1. **Scan Intake Agent** - Metadata extraction and validation
2. **Classification Agent** - Aneurysm detection (Yes/No)
3. **Segmentation Agent** - Precise aneurysm boundary detection
4. **Sizing Agent** - Dimensional measurements (length, width, area)
5. **Clinical Context Agent** - Patient history integration and prioritization
6. **Bias Detection Agent** - Algorithmic bias monitoring
7. **Explainability Agent** - AI decision explanations
8. **Risk Assessment Agent** - Rupture risk calculation
9. **Escalation Agent** - Urgent case notifications
10. **Billing Agent** - CPT code generation
11. **Clinical Trial Agent** - Trial matching and recommendations

##### **Specialized Agents**

- **Classification Agent** (`classification_agent.py`): Binary classification using TensorFlow/Keras model
- **Segmentation Agent** (`segmentation_agent.py`): U-Net based segmentation for precise localization
- **Sizing Agent** (`sizing_agent.py`): Automated measurement of aneurysm dimensions
- **Clinical Context Agent** (`clinical_context_agent.py`): Integrates patient history and prioritizes cases
- **Bias Detection Agent** (`bias_detection_agent.py`): Monitors prediction accuracy across demographics
- **Explainability Agent** (`explainability_agent.py`): Generates human-readable AI explanations
- **Risk Agent** (`risk_agent.py`): LLM-powered rupture risk assessment
- **Escalation Agent** (`escalation_agent.py`): Automated notification system for urgent cases
- **Billing Agent** (`billing_agent.py`): Generates appropriate medical billing codes
- **Scan Intake Agent** (`scan_intake_agent.py`): Validates and extracts scan metadata
- **Clinical Trial Agent** (`clinical_trial_agent.py`): Matches patients to relevant clinical trials
- **Preprocessing Agent** (`preprocessing_agent.py`): Image preprocessing and enhancement

#### 4. Machine Learning Models (`model/`)

- **Classification Model** (`classification_model.h5`, `classification_model.tflite`): Convolutional neural network for aneurysm detection
- **Segmentation Model** (`segmentation_model.h5`): U-Net architecture for precise segmentation

**📥 Model Download**: Download the pre-trained models from [Google Drive](https://drive.google.com/drive/folders/1H9LMq8HgVmcEqh9rj44RuCVA6DvCkPpt?usp=sharing)

## 📋 Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- SQLite (default) or PostgreSQL (production)
- GPU recommended for model inference

## 🛠️ Installation & Setup

### Option 1: Docker Deployment (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/muhammadasad149/Lilia-AI-Agentic-Backend.git
   cd Lilia-AI-Agentic-Backend
   ```

2. **Environment Configuration**
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_googel_api_key
   SECRET_KEY=your-secret-key-here
   ALGORITHM=HS256
   ACCESS_TOKEN_EXPIRE_MINUTES=720
   DATABASE_URL=sqlite:///./lilia_ai.db
   UPLOAD_DIR=./uploads
   ```

3. **Build and Run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

4. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Alternative Docs: http://localhost:8000/redoc

### Option 2: Local Development Setup

1. **Clone and navigate**
   ```bash
   git clone https://github.com/muhammadasad149/Lilia-AI-Agentic-Backend.git
   cd Lilia-AI-Agentic-Backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment variables**
   Create `.env` file as shown above

5. **Run the application**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## 🔧 API Usage

### Authentication

1. **Sign Up**
   ```bash
   curl -X POST "http://localhost:8000/signup" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "username=doctor1&password=securepass&email=doctor@example.com"
   ```

2. **Login**
   ```bash
   curl -X POST "http://localhost:8000/login" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "username=doctor1&password=securepass"
   ```

### Patient Management

- **Add Patient**: `POST /patients`
- **List Patients**: `GET /patients`
- **Get Patient**: `GET /patients/{patient_id}`
- **Update Patient**: `PUT /patients/{patient_id}`
- **Delete Patient**: `DELETE /patients/{patient_id}`

### Scan Processing

1. **Upload CT Scan**
   ```bash
   curl -X POST "http://localhost:8000/patients/{patient_id}/upload_scan" \
        -H "Authorization: Bearer {token}" \
        -F "file=@scan.nii"
   ```

2. **View Scan Results**
   ```bash
   curl -X GET "http://localhost:8000/patients/{patient_id}/scans" \
        -H "Authorization: Bearer {token}"
   ```

3. **Provide Feedback** (for bias monitoring)
   ```bash
   curl -X POST "http://localhost:8000/scans/{scan_id}/feedback" \
        -H "Authorization: Bearer {token}" \
        -H "Content-Type: application/json" \
        -d '{"actual_outcome": "true_positive"}'
   ```

## 🤖 AI Pipeline Details

### Processing Stages

1. **Scan Intake**: Validates NIfTI format, extracts metadata, quality assessment
2. **Classification**: CNN-based detection of AVM presence (confidence score)
3. **Segmentation**: U-Net segmentation if aneurysm detected
4. **Sizing**: Automated measurement of aneurysm dimensions in mm
5. **Clinical Context**: Patient history integration and case prioritization
6. **Bias Detection**: Logs predictions and monitors demographic performance
7. **Explainability**: Generates SHAP-based explanations and clinical reasoning
8. **Risk Assessment**: LLM-powered rupture risk calculation (STAT/URGENT/ROUTINE/LOW)
9. **Escalation**: Automated notifications for high-priority cases
10. **Billing**: CPT code generation based on findings
11. **Clinical Trials**: Matches patients to relevant research studies

### Output Structure

```json
{
  "scan_intake_agent": {...},
  "classification_agent": {
    "prediction": "Aneurysm Detected",
    "confidence": "95.67%"
  },
  "segmentation_agent": {...},
  "sizing_agent": {
    "summary": {
      "max_length_mm": 12.5,
      "max_width_mm": 8.3,
      "max_area_mm2": 78.2
    }
  },
  "clinical_context_agent": {
    "priority": "HIGH",
    "reason": "Large aneurysm with risk factors"
  },
  "bias_detection_agent": {...},
  "explainability_agent": "<html>explanation...</html>",
  "rupture_risk_agent": {
    "priority": "URGENT",
    "risk_score": 0.85
  },
  "escalation_tier": "Tier 2 - Expedited review required",
  "billing_optimization_agent": {...},
  "clinical_trials_matching_agent": {...},
  "scan_frames": ["base64_encoded_images..."],
  "pipeline_status": "completed"
}
```

## 📊 Models & Performance

### Classification Model
- **Architecture**: Convolutional Neural Network
- **Input**: CT scan slices (256x256)
- **Output**: Binary classification (Aneurysm/No Aneurysm)
- **Accuracy**: ~94% on validation set
- **Formats**: Keras (.h5) and TensorFlow Lite (.tflite)

### Segmentation Model
- **Architecture**: U-Net with attention mechanisms
- **Input**: CT scan slices
- **Output**: Pixel-wise segmentation masks
- **IoU Score**: ~0.87 on validation set
- **Format**: Keras (.h5)

## 🔒 Security & Compliance

- **JWT Authentication**: Secure token-based authentication
- **Role-based Access**: Doctor-patient data isolation
- **File Validation**: Strict NIfTI format validation
- **Audit Logging**: All predictions and user actions logged
- **Bias Monitoring**: Continuous algorithmic fairness assessment
- **HIPAA Considerations**: Designed for healthcare data handling

### Notebook Testing
- `app.ipynb`: Sizing agent experimentation and validation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠️ Disclaimer

This software is for research and development purposes. It is not intended for clinical use without proper validation, regulatory approval, and medical supervision. Always consult with qualified healthcare professionals for medical decisions.

---

**Built with ❤️ for advancing medical AI and improving patient outcomes**
