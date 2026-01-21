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

def generate_billing_codes(findings: dict, clinical_data: dict = None) -> dict:
    """
    Uses LLM to suggest ICD-10 and CPT codes based on aneurysm prediction and clinical context.

    Parameters:
    ----------
    findings : dict
        Dictionary containing model predictions (e.g., "prediction": "Aneurysm Detected").
    clinical_data : dict, optional
        Additional patient data for context-aware coding.

    Returns:
    -------
    dict
        Dictionary with suggested ICD-10 and CPT codes and explanation.
    """
    prompt = f"""
    You are a medical billing assistant AI. Based on the following patient findings and optional clinical data,
    suggest the most appropriate ICD-10 and CPT billing codes.

    Return only a Python dictionary in this format:
    {{
        "suggested_codes": {{
            "ICD-10": "<ICD-10 code>",
            "CPT": "<CPT code>"
        }},
        "explanation": "<brief explanation>"
    }}

    Patient Findings:
    {findings}

    Clinical Data:
    {clinical_data if clinical_data else 'N/A'}
    """

    try:
        response = llm.invoke(prompt)
        parsed = extract_dict_from_response(response.content)
        if parsed:
            return parsed
        else:
            return {
                "suggested_codes": {},
                "explanation": "LLM response was received but could not be parsed."
            }
    except Exception as e:
        return {
            "suggested_codes": {},
            "explanation": f"LLM failed: {str(e)}"
        }
