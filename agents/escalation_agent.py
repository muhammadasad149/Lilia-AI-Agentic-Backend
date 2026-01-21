def trigger_notification(urgency_tier: str, patient_info: dict, findings: dict) -> dict:
    """
    Simulates sending a notification to clinical staff based on urgency.
    """
    patient_id = patient_info.get('id', 'Unknown')
    finding_summary = findings.get('prediction', 'No findings')
    
    message = (
        f"🚨 {urgency_tier} ALERT\n"
        f"Patient ID: {patient_id}\n"
        f"Finding: {finding_summary}\n"
        f"Please review immediately."
    )
    
    print(f"PAGING SYSTEM: {message}")
    # Integrate with real notification systems here (e.g., Twilio, PagerDuty).
    
    return {
        "status": "Notification Sent",
        "urgency_tier": urgency_tier,
        "patient_id": patient_id,
        "message": message
    }