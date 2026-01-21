import os
import re
import ast
from langchain_google_genai import ChatGoogleGenerativeAI

# Configure Google API key
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def extract_dict_from_response(response_text: str) -> dict:
    """
    Extracts the first JSON/dict-like structure from LLM response safely.
    """
    try:
        # Match dictionary enclosed in triple or single quotes
        pattern = r"{.*}"
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            return ast.literal_eval(match.group())
    except Exception as e:
        print("Error parsing dict from response:", e)
    return {}

def calculate_rupture_risk_llm(aneurysm_details: dict, patient_history: dict, prediction_result: dict, scan_metadata: dict):
    """
    Use LLM to assess aneurysm rupture risk and suggest escalation priority.
    """
    prompt = f"""
    Patient Information:
    - Age: {patient_history.get('age', 'Unknown')}
    - Medical History: {patient_history.get('history', 'No history available')}
    - AI Prediction: {prediction_result.get('prediction', 'Unknown')}
    - Confidence: {prediction_result.get('confidence', 'Unknown')}
    - Scan Quality: {scan_metadata.get('quality_assessment', {}).get('overall_quality', 'Unknown')}

    Analyze and provide:
    - Risk stratification (Low, Moderate, High)
    - Priority level (STAT, URGENT, ROUTINE, LOW)
    - Clinical reasoning
    - Key risk factors
    - Recommendations

    Return result in JSON format like:
    {{
      "risk_level": "High",
      "priority": "STAT",
      "reasoning": "Elderly patient with hypertension and high-confidence aneurysm detection.",
      "key_risk_factors": ["Age > 60", "Hypertension", "Aneurysm Detected"],
      "recommendations": ["Immediate radiologist review", "Hospital admission"]
    }}
    """

    response = llm.invoke(prompt)
    risk_assessment = extract_dict_from_response(response.content)
    urgency = risk_assessment.get("priority", "Unknown")

    return {
        "escalation_tier": urgency,
        "risk_assessment": risk_assessment
    }
