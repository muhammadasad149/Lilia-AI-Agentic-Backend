from agents.classification_agent import model_predict
from agents.segmentation_agent import segmentation_pipeline
from agents.explainability_agent import EnhancedExplainabilityAgent
from agents.clinical_context_agent import EnhancedClinicalContextAgent
from agents.bias_detection_agent import BiasDetectionAgent
from agents.risk_agent import calculate_rupture_risk_llm
from agents.escalation_agent import trigger_notification
from agents.billing_agent import generate_billing_codes
from agents.scan_intake_agent import ScanIntakeAgent
from agents.clinical_trial_agent import ClinicalTrialMatchingAgent
from agents.sizing_agent import EnhancedSizingAgent  # Import the new sizing agent
from sqlalchemy.orm import Session
import nibabel as nib
import cv2
import base64
from io import BytesIO
from PIL import Image
import numpy as np

# Initialize agents
bias_agent = BiasDetectionAgent()
explainer_agent = EnhancedExplainabilityAgent()
scan_intake_agent = ScanIntakeAgent()
clinical_trial_agent = ClinicalTrialMatchingAgent()
clinical_context_agent = EnhancedClinicalContextAgent()
sizing_agent = EnhancedSizingAgent(pixel_to_mm_ratio=0.39)  # Initialize sizing agent


def convert_np_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(i) for i in obj]
    else:
        return obj


def extract_nii_frames(nii_path, img_size=(256, 256)):
    """Extract frames from NIfTI file and convert to base64"""
    nii_img = nib.load(nii_path)
    nii_data = nii_img.get_fdata()
    images_base64 = []
    for i in range(nii_data.shape[2]):
        slice_img = nii_data[:, :, i]
        normalized_slice = cv2.normalize(slice_img, None, 0, 255, cv2.NORM_MINMAX)
        resized_slice = cv2.resize(normalized_slice, img_size)
        img_uint8 = resized_slice.astype(np.uint8)
        img_pil = Image.fromarray(img_uint8)
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        images_base64.append(f"data:image/png;base64,{encoded_string}")
    return images_base64


def full_prediction_pipeline(nii_path: str, patient_id: int, db: Session):
    """
    Complete medical imaging analysis pipeline with all agents
    
    Pipeline Flow:
    1. Scan Intake Agent - Extract metadata and validate scan
    2. Classification Agent - Predict condition
    3. Segmentation Agent - Segment if aneurysm detected
    4. Sizing Agent - Measure aneurysm dimensions (NEW STAGE)
    5. Clinical Context Agent - Get patient context and prioritize
    6. Bias Detection Agent - Check for bias and log prediction
    7. Explainability Agent - Generate explanations
    8. Escalation Agent - Handle urgent cases
    9. Clinical Trial Matching Agent - Find relevant trials
    10. Billing Agent - Generate billing codes
    """
    
    # Stage 1: Scan Intake and Metadata Processing
    print("🔍 Stage 1: Scan Intake Processing...")
    scan_metadata = scan_intake_agent.process_scan(nii_path, patient_id)
    
    if 'error' in scan_metadata:
        return {
            'error': f"Scan intake failed: {scan_metadata['error']}",
            'stage': 'scan_intake'
        }
    
    # Check if scan is AI-ready
    if not scan_metadata.get('ai_processing_ready', {}).get('compatible', False):
        print("⚠️  Scan requires preprocessing for AI analysis")
        # In a real system, you might want to preprocess here
        # For now, we'll continue with a warning
    
    # Stage 2: AI Inference & Detection
    print("🤖 Stage 2: AI Classification...")
    try:
        class_label, confidence = model_predict(nii_path)
        confidence = float(confidence)
        print(f"Predicted Class: {class_label}, Confidence: {confidence:.2f}%")
        
        prediction_result = {
            "prediction": class_label,
            "confidence": f"{confidence:.2f}%"
        }
    except Exception as e:
        return {
            'error': f"AI prediction failed: {str(e)}",
            'stage': 'classification',
            'scan_metadata': scan_metadata
        }

    # Stage 3: Segmentation (if aneurysm detected)
    print("🎯 Stage 3: Segmentation Analysis...")
    segmentation_results = None
    if class_label == "Aneurysm Detected":
        try:
            segmentation_results = segmentation_pipeline(nii_path)
        except Exception as e:
            print(f"⚠️  Segmentation failed: {str(e)}")
            segmentation_results = {'error': f"Segmentation failed: {str(e)}"}

    # Stage 4: Sizing Analysis (NEW STAGE)
    print("📏 Stage 4: Aneurysm Sizing Analysis...")
    sizing_results = None
    if class_label == "Aneurysm Detected" and segmentation_results and 'error' not in segmentation_results:
        try:
            print("   Analyzing segmentation images for measurements...")
            sizing_results = sizing_agent.analyze_segmentation_results(segmentation_results)
            
            # Log sizing summary
            if 'summary' in sizing_results:
                summary = sizing_results['summary']
                print(f"   📊 Sizing Summary:")
                print(f"      Max Length: {summary.get('max_length_mm', 0)} mm")
                print(f"      Max Width: {summary.get('max_width_mm', 0)} mm")
                print(f"      Max Area: {summary.get('max_area_mm2', 0)} mm²")
                print(f"      Regions Detected: {summary.get('total_detected_regions', 0)}")
                print(f"      Slices with Detections: {summary.get('slices_with_detections', 0)}")
                
        except Exception as e:
            print(f"⚠️  Sizing analysis failed: {str(e)}")
            sizing_results = {'error': f"Sizing analysis failed: {str(e)}"}
    else:
        print("   Sizing skipped - no aneurysm detected or segmentation failed")
        sizing_results = {'skipped': 'No aneurysm detected or segmentation unavailable'}

    # Stage 5: Clinical Context Analysis
    print("🏥 Stage 5: Clinical Context Analysis...")
    try:
        clinical_data = clinical_context_agent.get_clinical_context(patient_id, db)
        
        # Include sizing data in prioritization if available
        enhanced_prediction = prediction_result.copy()
        if sizing_results and 'summary' in sizing_results:
            enhanced_prediction['sizing_summary'] = sizing_results['summary']
        
        priority_info = clinical_context_agent.prioritize_based_on_context(
            enhanced_prediction, clinical_data, scan_metadata
        )
        
        # Get additional treatment recommendations if aneurysm detected
        treatment_recommendations = None
        if class_label == "Aneurysm Detected":
            treatment_recommendations = clinical_context_agent.get_treatment_recommendations(
                clinical_data, enhanced_prediction
            )
            
    except Exception as e:
        print(f"⚠️  Clinical context retrieval failed: {str(e)}")
        clinical_data = {'error': f"Clinical context failed: {str(e)}"}
        priority_info = {'priority': 'Unknown', 'reason': 'Clinical data unavailable'}
        treatment_recommendations = None

    # Stage 6: Bias Detection and Logging
    print("⚖️  Stage 6: Bias Detection...")
    bias_analysis_report = {'error': 'Bias analysis unavailable'}
    try:
        if 'error' not in clinical_data:
            log_data = {
                "patient_id": patient_id,
                "prediction": class_label,
                "confidence": confidence,
                "age": clinical_data.get('age'),
                "gender": clinical_data.get('sex'),
                "ethnicity": clinical_data.get('ethnicity', 'Not Specified'),
                "scanner_model": scan_metadata.get('acquisition_info', {}).get('manufacturer_model', 'Unknown'),
                "hospital_id": clinical_data.get('hospital_id', 'Unknown')
            }
            bias_agent.log_prediction(db=db, **log_data)
            bias_analysis_report = bias_agent.run_bias_analysis(db=db, days_back=90)
    except Exception as e:
        print(f"⚠️  Bias detection failed: {str(e)}")
        bias_analysis_report = {'error': f"Bias analysis failed: {str(e)}"}

    # Stage 7: Explainability Analysis
    print("🔬 Stage 7: Explainability Analysis...")
    explainability_results = None
    try:
        # Include sizing data in explainability analysis
        enhanced_patient_data = clinical_data.copy() if isinstance(clinical_data, dict) and 'error' not in clinical_data else {}
        if sizing_results and 'summary' in sizing_results:
            enhanced_patient_data['aneurysm_measurements'] = sizing_results['summary']
        
        explainability_results = explainer_agent.run(
            nii_path=nii_path,
            prediction_data=enhanced_prediction,
            patient_data=enhanced_patient_data
        )
    except Exception as e:
        print(f"⚠️  Explainability analysis failed: {str(e)}")
        explainability_results = f"<div>Explainability analysis failed: {str(e)}</div>"

    # Stage 8: Escalation Management
    print("🚨 Stage 8: Escalation Assessment via LLM Risk Agent...")

    # ❶ Run the LLM risk agent with sizing data
    enhanced_aneurysm_details = segmentation_results.copy() if segmentation_results else {}
    if sizing_results and 'summary' in sizing_results:
        enhanced_aneurysm_details['sizing_measurements'] = sizing_results['summary']

    risk_assessment = calculate_rupture_risk_llm(
        aneurysm_details=enhanced_aneurysm_details,
        patient_history=clinical_data if isinstance(clinical_data, dict) else {},
        prediction_result=enhanced_prediction,
        scan_metadata=scan_metadata
    )

    # ❷ Map LLM priority → escalation tier
    priority_map = {
        "STAT":   "Tier 1 - Immediate radiologist notification",
        "URGENT": "Tier 2 - Expedited review required",
        "ROUTINE": "Tier 3 - Review when possible",
        "LOW":    "Tier 4 - Routine follow‑up"
    }
    urgency = priority_map.get(risk_assessment.get("priority", "ROUTINE"),
                            "Tier 3 - Review when possible")

    # ❸ Trigger paging for Tier 1 & Tier 2
    try:
        if urgency.startswith("Tier 1") or urgency.startswith("Tier 2"):
            # Include sizing info in notification
            enhanced_findings = prediction_result.copy()
            if sizing_results and 'summary' in sizing_results:
                enhanced_findings['measurements'] = sizing_results['summary']
            
            trigger_notification(
                urgency_tier=urgency,
                patient_info=clinical_data if isinstance(clinical_data, dict) else {},
                findings=enhanced_findings
            )
    except Exception as e:
        print(f"⚠️  Escalation processing failed: {str(e)}")
        risk_assessment["notification_error"] = str(e)

    # Stage 9: Billing Optimization
    print("💳 Stage 9: Billing Code Generation...")
    billing_codes = {}
    try:
        # Include sizing measurements in billing data
        enhanced_findings_for_billing = prediction_result.copy()
        if sizing_results and 'summary' in sizing_results:
            enhanced_findings_for_billing['aneurysm_size'] = sizing_results['summary']
        
        billing_codes = generate_billing_codes(
            findings=enhanced_findings_for_billing,
            clinical_data=clinical_data if isinstance(clinical_data, dict) else {}
        )
    except Exception as e:
        print(f"⚠️  Billing code generation failed: {str(e)}")
        billing_codes = {"error": f"Billing code generation failed: {str(e)}"}

    # Stage 10: Clinical Trial Matching
    print("🧬 Stage 10: Clinical Trial Matching...")
    clinical_trial_results = {'error': 'Clinical trial matching unavailable'}
    try:
        if 'error' not in clinical_data and class_label == "Aneurysm Detected":
            # Include sizing data for trial matching
            enhanced_patient_data_for_trials = clinical_data.copy()
            if sizing_results and 'summary' in sizing_results:
                enhanced_patient_data_for_trials['aneurysm_size'] = sizing_results['summary']
            
            # Only search for trials if aneurysm is detected
            matched_trials = clinical_trial_agent.find_matching_trials(
                prediction_result=enhanced_prediction,
                patient_data=enhanced_patient_data_for_trials,
                max_results=5
            )
            clinical_trial_results = clinical_trial_agent.get_trial_summary(matched_trials)
            
            # Generate detailed report
            trial_report = clinical_trial_agent.generate_trial_report(
                matched_trials, enhanced_patient_data_for_trials
            )
            clinical_trial_results['detailed_report'] = trial_report
            
    except Exception as e:
        print(f"⚠️  Clinical trial matching failed: {str(e)}")
        clinical_trial_results = {'error': f"Clinical trial matching failed: {str(e)}"}

    # Stage 11: Final Processing and Output
    print("📊 Stage 11: Final Processing...")
    try:
        scan_frames = extract_nii_frames(nii_path)
    except Exception as e:
        print(f"⚠️  Frame extraction failed: {str(e)}")
        scan_frames = []

    # Consolidate all results
    final_output = {
        "scan_intake_agent": scan_metadata,
        "classification_agent": prediction_result,
        "segmentation_agent": segmentation_results,
        "sizing_agent": sizing_results,  # NEW: Add sizing results
        "bias_detection_agent": bias_analysis_report,
        "explainability_agent": explainability_results,
        "clinical_context_agent": priority_info,
        "treatment_recommendations": treatment_recommendations,
        "rupture_risk_agent": risk_assessment,
        "escalation_tier": urgency,
        "billing_optimization_agent": billing_codes,
        "clinical_trials_matching_agent": clinical_trial_results,
        "scan_frames": scan_frames,
        "pipeline_status": "completed",
        "processing_timestamp": scan_metadata.get('processing_timestamp')
    }

    print("✅ Pipeline completed successfully!")
    return convert_np_types(final_output)


def get_pipeline_status_summary(pipeline_result: dict) -> str:
    """Generate a human-readable summary of the pipeline execution"""
    
    if 'error' in pipeline_result:
        return f"❌ Pipeline failed at {pipeline_result.get('stage', 'unknown')} stage: {pipeline_result['error']}"
    
    summary = "🏥 MEDICAL IMAGING ANALYSIS SUMMARY\n"
    summary += "=" * 50 + "\n\n"
    
    # Scan info
    scan_meta = pipeline_result.get('scan_intake_agent', {})
    if scan_meta:
        summary += f"📁 Scan: {scan_meta.get('file_format', 'Unknown')} format\n"
        summary += f"📏 Size: {scan_meta.get('file_size_mb', 'Unknown')} MB\n"
        quality = scan_meta.get('quality_assessment', {}).get('overall_quality', 'Unknown')
        summary += f"✅ Quality: {quality}\n\n"
    
    # Prediction results
    pred_result = pipeline_result.get('classification_agent', {})
    if pred_result:
        summary += f"🤖 AI Prediction: {pred_result.get('prediction', 'Unknown')}\n"
        summary += f"📊 Confidence: {pred_result.get('confidence', 'Unknown')}\n\n"
    
    # NEW: Sizing results summary
    sizing_result = pipeline_result.get('sizing_agent', {})
    if sizing_result and 'summary' in sizing_result:
        sizing_summary = sizing_result['summary']
        summary += "📏 ANEURYSM MEASUREMENTS:\n"
        summary += f"   Max Length: {sizing_summary.get('max_length_mm', 0)} mm\n"
        summary += f"   Max Width: {sizing_summary.get('max_width_mm', 0)} mm\n"
        summary += f"   Max Area: {sizing_summary.get('max_area_mm2', 0)} mm²\n"
        summary += f"   Total Regions: {sizing_summary.get('total_detected_regions', 0)}\n"
        summary += f"   Slices Analyzed: {sizing_summary.get('total_slices_analyzed', 0)}\n\n"
    elif sizing_result and 'error' in sizing_result:
        summary += f"📏 Sizing: Failed - {sizing_result['error']}\n\n"
    elif sizing_result and 'skipped' in sizing_result:
        summary += "📏 Sizing: Skipped - No aneurysm detected\n\n"
    
    # Priority and escalation
    priority = pipeline_result.get('clinical_context_agent', {}).get('priority', 'Unknown')
    escalation = pipeline_result.get('escalation_tier', 'Unknown')
    summary += f"⚠️  Priority: {priority}\n"
    summary += f"🚨 Escalation: {escalation}\n\n"
    
    # Clinical trials
    trials = pipeline_result.get('clinical_trials_matching_agent', {})
    if trials and 'error' not in trials:
        trial_count = trials.get('total_matches', 0)
        summary += f"🧬 Clinical Trials: {trial_count} matches found\n"
        if trial_count > 0:
            top_match = trials.get('top_match', {})
            summary += f"   Top Match: {top_match.get('title', 'Unknown')} ({top_match.get('match_score', 0):.1f}%)\n"
    
    summary += f"\n⏰ Processed: {pipeline_result.get('processing_timestamp', 'Unknown')}\n"
    
    return summary


def get_sizing_detailed_report(sizing_results: dict) -> str:
    """Generate a detailed report of sizing analysis"""
    
    if not sizing_results:
        return "No sizing data available."
    
    if 'error' in sizing_results:
        return f"Sizing Analysis Failed: {sizing_results['error']}"
    
    if 'skipped' in sizing_results:
        return "Sizing Analysis Skipped: No aneurysm detected or segmentation unavailable."
    
    report = "📏 DETAILED ANEURYSM SIZING REPORT\n"
    report += "=" * 40 + "\n\n"
    
    # Summary section
    if 'summary' in sizing_results:
        summary = sizing_results['summary']
        report += "📊 SUMMARY MEASUREMENTS:\n"
        report += f"Maximum Length: {summary.get('max_length_mm', 0)} mm\n"
        report += f"Maximum Width: {summary.get('max_width_mm', 0)} mm\n"
        report += f"Maximum Area: {summary.get('max_area_mm2', 0)} mm²\n"
        report += f"Average Length: {summary.get('avg_length_mm', 0)} mm\n"
        report += f"Average Width: {summary.get('avg_width_mm', 0)} mm\n"
        report += f"Average Area: {summary.get('avg_area_mm2', 0)} mm²\n\n"
        
        report += "📈 DETECTION STATISTICS:\n"
        report += f"Total Detected Regions: {summary.get('total_detected_regions', 0)}\n"
        report += f"Slices with Detections: {summary.get('slices_with_detections', 0)}\n"
        report += f"Total Slices Analyzed: {summary.get('total_slices_analyzed', 0)}\n"
        detection_rate = (summary.get('slices_with_detections', 0) / 
                         max(summary.get('total_slices_analyzed', 1), 1)) * 100
        report += f"Detection Rate: {detection_rate:.1f}%\n\n"
    
    # Per-slice details
    if 'slice_measurements' in sizing_results:
        measurements = sizing_results['slice_measurements']
        if measurements:
            report += "🔍 PER-SLICE MEASUREMENTS:\n"
            report += "-" * 30 + "\n"
            
            for measurement in measurements:
                slice_idx = measurement.get('slice_index', 'Unknown')
                slice_num = measurement.get('slice_number', 'Unknown')
                
                report += f"Slice #{slice_num} (Index: {slice_idx}):\n"
                
                if measurement.get('total_regions', 0) > 0:
                    report += f"  Length: {measurement.get('length_mm', 0)} mm\n"
                    report += f"  Width: {measurement.get('width_mm', 0)} mm\n"
                    report += f"  Area: {measurement.get('area_mm2', 0)} mm²\n"
                    report += f"  Regions: {measurement.get('total_regions', 0)}\n"
                    
                    # Individual region details if available
                    if measurement.get('measurements_per_region'):
                        for region in measurement['measurements_per_region']:
                            region_id = region.get('region_id', 'Unknown')
                            report += f"    Region {region_id}: {region.get('length_mm', 0)}×{region.get('width_mm', 0)} mm, {region.get('area_mm2', 0)} mm²\n"
                else:
                    report += "  No aneurysm detected\n"
                
                report += "\n"
    
    return report


