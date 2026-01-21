import os
import json
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from fastapi import HTTPException
from datetime import datetime
from models import Patient
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import re


load_dotenv()

# Configure Google Generative AI via Langchain
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=1200,
    timeout=30,
    max_retries=2,
)


def extract_json_dict_from_response(response_content: str):
    """
    Extract JSON object (dict) from a string using regex.
    Removes markdown formatting like ```json and parses only the dict.
    """
    # Regex to find JSON object (non-greedy match between curly braces)
    match = re.search(r'\{.*\}', response_content, re.DOTALL)
    
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print("❌ JSON decode error:", e)
            print("🔍 JSON String:", json_str)
            return None
    else:
        print("❌ No JSON object found in the response.")
        return None


class EnhancedClinicalContextAgent:
    """
    Enhanced Clinical Context Agent using Google Generative AI for analysis.
    """
    def __init__(self):
        self.context_prompts = self._load_clinical_prompts()

    def _load_clinical_prompts(self) -> Dict[str, str]:
        return {
            "treatment_readiness": """
            You are a perioperative medicine AI consultant.

            Patient Data:
            - Age: {age}
            - Medical History: {history}
            - Current Diagnosis: {prediction}
            - AI Confidence: {confidence}

            Assess treatment readiness:
            1. Surgical candidacy assessment
            2. Anesthesia risk factors
            3. Required preoperative workup
            4. Optimization recommendations
            5. Alternative treatment considerations

            Respond in structured JSON.
            """
        }

    def get_clinical_context(self, patient_id: int, db: Session) -> Dict[str, Any]:
        patient = db.query(Patient).filter(Patient.id == patient_id).first()
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient with ID {patient_id} not found in database.")
        return {
            "patient_id": patient.id,
            "age": patient.age,
            "sex": getattr(patient, 'sex', 'Unknown'),
            "history": patient.history,
            "hospital_id": getattr(patient, 'hospital_id', 'Unknown'),
            "ethnicity": getattr(patient, 'ethnicity', 'Not Specified'),
            "retrieved_at": datetime.now().isoformat()
        }

    def analyze_clinical_context_with_ai(self, clinical_data: Dict[str, Any],
                                         prediction_result: Dict[str, Any],
                                         scan_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

        try:
            prediction = prediction_result.get('prediction', '').lower()
            prompt_key = "aneurysm_risk_assessment" if 'aneurysm' in prediction else "prioritization_analysis"

            prompt = f"""
            You are a clinical decision support AI specializing in cerebrovascular conditions. Analyze the provided patient data and return a structured JSON response that includes aneurysm risk assessment, prioritization, comorbidity analysis, and treatment readiness.

            Patient Information:
            - Age: {clinical_data.get('age', 'Unknown')}
            - Medical History: {clinical_data.get('history', 'No history available')}
            - AI Prediction: {prediction_result.get('prediction', 'Unknown')}
            - Confidence: {prediction_result.get('confidence', 'Unknown')}
            - Scan Quality: {scan_metadata.get('quality_assessment', {}).get('overall_quality', 'Unknown') if scan_metadata else 'Unknown'}

            Analyze and provide:
            - Risk stratification
            - Priority level (STAT, URGENT, ROUTINE, LOW)
            - Clinical reasoning
            - Key risk factors
            - Recommendations

            Return the result in the following JSON format:

            {{
                "aneurysm_risk_assessment": {{
                    "patient_info": {{
                        "age": "Unknown",
                        "history": "No history available",
                        "prediction": "Unknown",
                        "confidence": "Unknown"
                    }},
                    "instructions": {{
                        "output_fields": [
                            "Risk stratification",
                            "Key risk factors",
                            "Urgency level",
                            "Clinical considerations",
                            "Immediate actions"
                        ],
                        "considerations": [
                            "Age-related risk",
                            "Comorbidities",
                            "Family history",
                            "Aneurysm characteristics"
                        ],
                        "reference": "Clinical guidelines for aneurysm management"
                    }}
                }},
                "prioritization_analysis": {{
                    "patient_info": {{
                        "age": "Unknown",
                        "history": "No history available",
                        "prediction": "Unknown",
                        "confidence": "Unknown",
                        "scan_quality": "Unknown"
                    }},
                    "instructions": {{
                        "priority_levels": [
                            "STAT", "URGENT", "ROUTINE", "LOW"
                        ],
                        "considerations": [
                            "Urgency of findings",
                            "Patient risk profile",
                            "Likelihood of deterioration",
                            "Efficient triage"
                        ],
                        "output": "Priority justification and recommendation"
                    }}
                }},
                "comorbidity_analysis": {{
                    "patient_profile": {{
                        "age": "Unknown",
                        "history": "No history available",
                        "finding": "Unknown"
                    }},
                    "instructions": {{
                        "analyze": [
                            "Relevant comorbidities",
                            "Contraindications",
                            "Risk factors",
                            "Precautions",
                            "Further evaluations"
                        ],
                        "focus_on": [
                            "Treatment modification",
                            "Surgical risk",
                            "Prognostic impact"
                        ],
                        "format": "Structured clinical insights"
                    }}
                }},
                "treatment_readiness": {{
                    "patient_data": {{
                        "age": "Unknown",
                        "history": "No history available",
                        "diagnosis": "Unknown",
                        "confidence": "Unknown"
                    }},
                    "instructions": {{
                        "assess": [
                            "Surgical candidacy",
                            "Anesthesia risks",
                            "Preoperative requirements",
                            "Optimization strategies",
                            "Alternative treatments"
                        ],
                        "consider": [
                            "Age and comorbidities",
                            "Cardio-pulmonary function",
                            "Neurological state"
                        ],
                        "output": "Comprehensive readiness analysis"
                    }}
                }}
            }}
            """


            messages = [
                ("system", "You are a clinical decision support AI."),
                ("human", prompt)
            ]
            response = llm.invoke(messages)
            print(f"AI analysis response: {response}")
            
            raw_response = response.content  # from llm.invoke(...)
            analysis = extract_json_dict_from_response(raw_response)

            return {
                "ai_analysis": analysis,
                "clinical_data": clinical_data,
                "analysis_timestamp": datetime.now().isoformat(),
                "prompt_type": prompt_key
            }

        except Exception as e:
            print(f"AI analysis failed: {str(e)}")
            return self._fallback_analysis(clinical_data, prediction_result)

    def _fallback_analysis(self, clinical_data: Dict[str, Any], prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        analysis = {
            "risk_stratification": "Medium",
            "priority_level": "ROUTINE",
            "key_factors": [],
            "recommendations": [],
            "clinical_data": clinical_data,
            "analysis_type": "rule_based_fallback",
            "analysis_timestamp": datetime.now().isoformat()
        }

        age = clinical_data.get('age', 0)
        history = clinical_data.get('history', '').lower()
        prediction = prediction_result.get('prediction', '').lower()

        if 'aneurysm' in prediction:
            analysis["risk_stratification"] = "High"
            analysis["priority_level"] = "URGENT"
            analysis["key_factors"].append("Brain aneurysm detected")
            if age > 60:
                analysis["risk_stratification"] = "Critical"
                analysis["priority_level"] = "STAT"
                analysis["key_factors"].append("Advanced age")
            if any(term in history for term in ['hypertension', 'smoking', 'family history']):
                analysis["key_factors"].append("Cardiovascular risk factors")
            analysis["recommendations"].extend([
                "Neurosurgical consultation",
                "Consider CTA/MRA",
                "Monitor blood pressure"
            ])

        return analysis

    def prioritize_based_on_context(self, prediction_result: Dict[str, Any],
                                    clinical_data: Dict[str, Any],
                                    scan_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ai_analysis = self.analyze_clinical_context_with_ai(clinical_data, prediction_result, scan_metadata)

        if "ai_analysis" in ai_analysis:
            ai_data = ai_analysis["ai_analysis"]
            ai_priority = ai_data.get("priority_level", "ROUTINE")
            ai_risk = ai_data.get("risk_stratification", "Medium")
            recommendations = ai_data.get("recommendations", [])
        else:
            ai_priority = ai_analysis.get("priority_level", "ROUTINE")
            ai_risk = ai_analysis.get("risk_stratification", "Medium")
            recommendations = ai_analysis.get("recommendations", [])

        priority_map = {"STAT": "Critical", "URGENT": "High", "ROUTINE": "Standard", "LOW": "Low"}

        return {
            "priority": priority_map.get(ai_priority, "Standard"),
            "ai_priority_level": ai_priority,
            "risk_stratification": ai_risk,
            "recommendations": recommendations,
            "clinical_reasoning": ai_data.get("clinical_considerations", "Not available"),
            "key_factors": ai_data.get("key_risk_factors", ai_analysis.get("key_factors", [])),
            "contextual_data": clinical_data,
            "ai_analysis_summary": ai_analysis,
            "priority_timestamp": datetime.now().isoformat()
        }

    def get_treatment_recommendations(self, clinical_data: Dict[str, Any], prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = self.context_prompts["treatment_readiness"].format(
                age=clinical_data.get('age', 'Unknown'),
                history=clinical_data.get('history', 'No history available'),
                prediction=prediction_result.get('prediction', 'Unknown'),
                confidence=prediction_result.get('confidence', 'Unknown')
            )

            messages = [
                ("system", "You are a clinical treatment planning AI."),
                ("human", prompt)
            ]

            response = llm.invoke(messages)
            # result = json.loads(response)
            result = extract_json_dict_from_response(response.content)


            return {
                "treatment_recommendations": result,
                "analysis_timestamp": datetime.now().isoformat(),
                "patient_id": clinical_data.get('patient_id')
            }

        except Exception as e:
            return {"error": f"Treatment analysis failed: {str(e)}"}

    def generate_clinical_summary(self, priority_result: Dict[str, Any]) -> str:
        clinical_data = priority_result.get('contextual_data', {})
        summary = (
            f"🏥 CLINICAL CONTEXT ANALYSIS\n{'=' * 40}\n\n"
            f"👤 Patient ID: {clinical_data.get('patient_id', 'Unknown')}\n"
            f"📅 Age: {clinical_data.get('age', 'Unknown')}\n"
            f"📋 History: {clinical_data.get('history', 'No history available')}\n\n"
            f"🚨 Priority Level: {priority_result.get('priority', 'Unknown')}\n"
            f"⚠️  Risk Level: {priority_result.get('risk_stratification', 'Unknown')}\n"
            f"🤖 AI Priority: {priority_result.get('ai_priority_level', 'Unknown')}\n\n"
        )

        key_factors = priority_result.get('key_factors', [])
        if key_factors:
            summary += "🔍 Key Risk Factors:\n"
            summary += "\n".join(f"   • {kf}" for kf in key_factors) + "\n\n"

        recommendations = priority_result.get('recommendations', [])
        if recommendations:
            summary += "💡 Recommendations:\n"
            summary += "\n".join(f"   • {rec}" for rec in recommendations) + "\n\n"

        summary += f"🧠 Clinical Reasoning:\n{priority_result.get('clinical_reasoning', 'None')}\n\n"
        summary += f"⏰ Analysis Time: {priority_result.get('priority_timestamp', 'Unknown')}\n"

        return summary


# Backward compatibility helpers
def get_clinical_context(patient_id: int, db: Session):
    return EnhancedClinicalContextAgent().get_clinical_context(patient_id, db)

def prioritize_based_on_context(prediction_result, clinical_data, scan_metadata=None):
    return EnhancedClinicalContextAgent().prioritize_based_on_context(prediction_result, clinical_data, scan_metadata)
