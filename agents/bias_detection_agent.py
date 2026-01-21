import pandas as pd
import json
from datetime import datetime, timedelta
from typing import List, Dict
from sqlalchemy.orm import Session
from models import PredictionsDemographics, BiasMetrics, BiasAlerts
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

# Configure LLM
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

@dataclass
class BiasMetricsData:
    demographic_group: str
    total_predictions: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    false_negative_rate: float

class BiasDetectionAgent:
    def __init__(self):
        self.llm = llm if llm else None
        self.bias_thresholds = {
            'accuracy_disparity': 0.05,
            'fpr_disparity': 0.05
        }
        # Minimum number of samples required for reliable bias analysis
        self.min_samples_per_group = 30
        self.min_samples_each_class = 5  # Minimum samples of each outcome class

    def log_prediction(self, db: Session, **kwargs):
        prediction_log = PredictionsDemographics(**kwargs)
        db.add(prediction_log)
        db.commit()
        print(f"[LOG] Logged prediction for patient_id={kwargs.get('patient_id')} | prediction={kwargs.get('prediction')}")

    def update_actual_outcome(self, db: Session, patient_id: int, actual_outcome: str):
        prediction_log = db.query(PredictionsDemographics).filter(PredictionsDemographics.patient_id == patient_id).first()
        if prediction_log:
            prediction_log.actual_outcome = actual_outcome
            db.commit()
            print(f"[LOG] Updated actual_outcome='{actual_outcome}' for patient_id={patient_id}")
        else:
            print(f"[WARN] No prediction log found for patient_id={patient_id}")

    def calculate_metrics(self, predictions: List[str], actual_outcomes: List[str]) -> Dict:
        """Calculate bias metrics with proper error handling for edge cases."""
        pred_binary = [1 if p == "Aneurysm Detected" else 0 for p in predictions]
        actual_binary = [1 if a == "Aneurysm Detected" else 0 for a in actual_outcomes]
        
        # Check if we have both classes in actual outcomes
        unique_actual = set(actual_binary)
        unique_pred = set(pred_binary)
        
        if len(unique_actual) < 2:
            print(f"[WARN] Only one class found in actual outcomes: {unique_actual}")
            return {}
            
        # Use labels parameter to ensure proper confusion matrix shape
        try:
            cm = confusion_matrix(actual_binary, pred_binary, labels=[0, 1])
            
            if cm.shape != (2, 2):
                print(f"[WARN] Unexpected confusion matrix shape: {cm.shape}")
                return {}

            tn, fp, fn, tp = cm.ravel()
            print(f"[DEBUG] Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            
            # Check for edge cases that would cause division by zero
            if (fp + tn) == 0:
                fpr = 0
                print("[WARN] No negative cases found, FPR set to 0")
            else:
                fpr = fp / (fp + tn)
                
            if (fn + tp) == 0:
                fnr = 0
                print("[WARN] No positive cases found, FNR set to 0")
            else:
                fnr = fn / (fn + tp)
            
            return {
                'accuracy': accuracy_score(actual_binary, pred_binary),
                'precision': precision_score(actual_binary, pred_binary, zero_division=0),
                'recall': recall_score(actual_binary, pred_binary, zero_division=0),
                'f1_score': f1_score(actual_binary, pred_binary, zero_division=0),
                'false_positive_rate': fpr,
                'false_negative_rate': fnr
            }
            
        except Exception as e:
            print(f"[ERROR] Error calculating metrics: {str(e)}")
            return {}

    def check_data_sufficiency(self, subset_df: pd.DataFrame) -> tuple[bool, str]:
        """Check if the data subset has sufficient samples for reliable bias analysis."""
        total_samples = len(subset_df)
        
        if total_samples < self.min_samples_per_group:
            return False, f"Insufficient total samples ({total_samples} < {self.min_samples_per_group})"
        
        # Check distribution of actual outcomes
        outcome_counts = subset_df['actual_outcome'].value_counts()
        
        if len(outcome_counts) < 2:
            return False, f"Only one outcome class present: {outcome_counts.index.tolist()}"
        
        min_class_count = outcome_counts.min()
        if min_class_count < self.min_samples_each_class:
            return False, f"Insufficient samples in minority class ({min_class_count} < {self.min_samples_each_class})"
        
        return True, "Sufficient data for analysis"

    def analyze_demographic_bias(self, db: Session, days_back: int = 30) -> List[BiasMetricsData]:
        start_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        print(f"[LOG] Analyzing predictions since {start_date}")
        
        query = db.query(PredictionsDemographics).filter(
            PredictionsDemographics.actual_outcome != None,
            PredictionsDemographics.timestamp >= start_date
        ).statement
        
        df = pd.read_sql(query, db.bind)

        print(f"[LOG] Total rows fetched: {len(df)}")
        if df.empty:
            print("[WARN] No prediction logs with actual outcomes found.")
            return []

        # Check overall data sufficiency
        overall_sufficient, overall_msg = self.check_data_sufficiency(df)
        if not overall_sufficient:
            print(f"[WARN] Overall dataset insufficient for bias analysis: {overall_msg}")
            print("[INFO] Bias analysis requires more historical data with actual outcomes")
            return []

        df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 80, 120], labels=['<40', '40-60', '60-80', '80+'])
        demographic_columns = ['gender', 'age_group', 'ethnicity', 'scanner_model']
        all_metrics = []

        for col in demographic_columns:
            print(f"[INFO] Checking bias for: {col}")
            for group_value in df[col].unique():
                if pd.isna(group_value):
                    print(f"[INFO] Skipping NaN value in {col}")
                    continue

                subset = df[df[col] == group_value]
                print(f"[INFO] Group '{col} = {group_value}' has {len(subset)} records")

                # Check data sufficiency for this group
                sufficient, msg = self.check_data_sufficiency(subset)
                if not sufficient:
                    print(f"[INFO] Skipping group '{group_value}': {msg}")
                    continue

                metrics = self.calculate_metrics(
                    subset['prediction'].tolist(),
                    subset['actual_outcome'].tolist()
                )

                if metrics:
                    all_metrics.append(
                        BiasMetricsData(
                            demographic_group=f"{col.replace('_', ' ').title()}: {group_value}",
                            total_predictions=len(subset),
                            **metrics
                        )
                    )
                    print(f"[SUCCESS] Bias metrics calculated for group: {group_value}")
                else:
                    print(f"[WARN] Metrics calculation failed for group: {group_value}")

        print(f"[LOG] Total bias metric groups generated: {len(all_metrics)}")
        
        if len(all_metrics) == 0:
            print("[INFO] No demographic groups had sufficient data for bias analysis")
            print(f"[INFO] Requirements: {self.min_samples_per_group} total samples, {self.min_samples_each_class} per outcome class")
        
        return all_metrics
    
    def generate_llm_bias_summary(self, bias_metrics: List[BiasMetricsData]) -> str:
        if not self.llm:
            return "[INFO] No LLM integrated. Skipping summary generation."

        if not bias_metrics:
            return "[INFO] No bias metrics available for summary generation. Insufficient data for analysis."

        # Prepare the prompt content
        metrics_for_llm = [m.__dict__ for m in bias_metrics]
        prompt = f"""
        You are a healthcare AI ethics assistant. Summarize the following bias analysis:
        {json.dumps(metrics_for_llm, indent=2)}

        Provide:
        - A plain-language summary of any disparities detected
        - Which demographic groups are most affected
        - Possible ethical implications
        - Recommendations to address these biases

        Respond only in plain English.
        """

        try:
            # Call the LLM
            response = self.llm.invoke([
                ("system", "You are a fairness auditing assistant."),
                ("human", prompt)
            ])
            return response.content.strip()
        except Exception as e:
            return f"[ERROR] LLM summary generation failed: {str(e)}"

    def run_bias_analysis(self, db: Session, days_back: int = 30) -> Dict:
        print("[LOG] Running bias analysis...")
        bias_metrics_data = self.analyze_demographic_bias(db, days_back)
        
        result = {
            'analysis_timestamp': datetime.now().isoformat(),
            'bias_metrics': [m.__dict__ for m in bias_metrics_data],
            'data_requirements': {
                'min_samples_per_group': self.min_samples_per_group,
                'min_samples_each_class': self.min_samples_each_class
            }
        }
        
        # Generate summary
        summary = self.generate_llm_bias_summary(bias_metrics_data)
        result["llm_summary"] = summary
        
        # Add recommendations if insufficient data
        if not bias_metrics_data:
            result["recommendations"] = [
                "Collect more prediction data with actual outcomes",
                f"Ensure at least {self.min_samples_per_group} samples per demographic group",
                f"Ensure both 'Aneurysm Detected' and 'No Aneurysm' outcomes are present",
                "Consider extending the analysis timeframe to gather more data"
            ]

        return result