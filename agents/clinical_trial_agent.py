import requests
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
from dataclasses import dataclass


@dataclass
class ClinicalTrialMatch:
    """Data class for clinical trial match results"""
    nct_id: str
    title: str
    status: str
    phase: str
    condition: str
    location: str
    eligibility_criteria: str
    contact_info: str
    match_score: float
    match_reasons: List[str]


class ClinicalTrialMatchingAgent:
    """
    Agent for matching patients with relevant clinical trials using ClinicalTrials.gov API v2
    """
    
    def __init__(self):
        # Updated to use the new API v2 endpoint
        self.base_url = "https://clinicaltrials.gov/api/v2/studies"
        self.headers = {
            'User-Agent': 'MedicalAI-System/1.0',
            'Accept': 'application/json'
        }
        
        # Condition mappings for brain aneurysm related terms
        self.condition_mappings = {
            'aneurysm': ['Brain Aneurysm', 'Cerebral Aneurysm', 'Intracranial Aneurysm', 
                        'Subarachnoid Hemorrhage', 'Aneurysm'],
            'stroke': ['Stroke', 'Cerebrovascular Disease', 'Brain Ischemia'],
            'hemorrhage': ['Subarachnoid Hemorrhage', 'Intracerebral Hemorrhage', 'Brain Hemorrhage'],
            'vascular': ['Cerebrovascular Disease', 'Vascular Malformation', 'Arteriovenous Malformation']
        }
        
        # Age group mappings
        self.age_groups = {
            'pediatric': (0, 17),
            'adult': (18, 64),
            'elderly': (65, 100)
        }
    
    def search_trials(self, condition: str, location: str = "United States", 
                     age: Optional[int] = None, max_results: int = 10) -> List[Dict]:
        """Search for clinical trials using ClinicalTrials.gov API v2"""
        
        # Build search expression for the new API
        search_expr = f"AREA[Condition]{condition}"
        
        # Add location filter
        if location:
            search_expr += f" AND AREA[LocationCountry]{location}"
        
        # Add recruiting status
        search_expr += " AND AREA[OverallStatus]Recruiting"
        
        # Build parameters for the new API format
        params = {
            'format': 'json',
            'fields': 'NCTId,BriefTitle,OverallStatus,Phase,Condition,LocationFacility,EligibilityCriteria,CentralContactName,CentralContactPhone,CentralContactEMail,LocationCity,LocationState,LocationCountry,LocationContactName,LocationContactPhone,LocationContactEMail',
            'query.cond': condition,
            'query.locn': location,
            'query.recrs': 'Open',  # Recruiting studies
            'pageSize': max_results
        }
        
        # Add age filter if provided
        if age:
            if age < 18:
                params['query.age'] = '0-17'
            elif age >= 65:
                params['query.age'] = '65-'
            else:
                params['query.age'] = '18-64'
        
        try:
            response = requests.get(self.base_url, params=params, headers=self.headers, timeout=15)
            
            # Print debug information
            print(f"Request URL: {response.url}")
            print(f"Response Status: {response.status_code}")
            
            if response.status_code == 400:
                print(f"Response content: {response.text}")
                # Try with simplified parameters
                return self._search_trials_simple(condition, location, age, max_results)
            
            response.raise_for_status()
            data = response.json()
            return data.get('studies', [])
            
        except requests.RequestException as e:
            print(f"Error searching clinical trials: {e}")
            # Fallback to simple search
            return self._search_trials_simple(condition, location, age, max_results)
    
    def _search_trials_simple(self, condition: str, location: str, age: Optional[int], max_results: int) -> List[Dict]:
        """Simplified search as fallback"""
        try:
            # Very basic parameters
            params = {
                'format': 'json',
                'query.cond': condition,
                'pageSize': max_results
            }
            
            response = requests.get(self.base_url, params=params, headers=self.headers, timeout=15)
            print(f"Fallback request URL: {response.url}")
            print(f"Fallback response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                return data.get('studies', [])
            else:
                print(f"Fallback response content: {response.text}")
                return []
                
        except requests.RequestException as e:
            print(f"Fallback search also failed: {e}")
            return []
    
    def extract_trial_details(self, trial_data: Dict) -> Dict[str, Any]:
        """Extract and structure trial details from API response"""
        
        protocol_section = trial_data.get('protocolSection', {})
        identification = protocol_section.get('identificationModule', {})
        status = protocol_section.get('statusModule', {})
        conditions = protocol_section.get('conditionsModule', {})
        eligibility = protocol_section.get('eligibilityModule', {})
        contacts = protocol_section.get('contactsModule', {})
        
        # Extract location information
        locations = []
        if 'locationsModule' in protocol_section:
            for location in protocol_section['locationsModule'].get('locations', []):
                facility = location.get('facility', {})
                locations.append({
                    'facility': facility.get('name', ''),
                    'city': location.get('city', ''),
                    'state': location.get('state', ''),
                    'country': location.get('country', ''),
                    'status': location.get('status', '')
                })
        
        # Extract contact information
        contact_info = {}
        if 'centralContacts' in contacts:
            for contact in contacts['centralContacts']:
                contact_info = {
                    'name': contact.get('name', ''),
                    'phone': contact.get('phone', ''),
                    'email': contact.get('email', '')
                }
                break  # Take first contact
        
        return {
            'nct_id': identification.get('nctId', ''),
            'title': identification.get('briefTitle', ''),
            'status': status.get('overallStatus', ''),
            'phase': status.get('phase', 'Not Specified'),
            'conditions': conditions.get('conditions', []),
            'eligibility_criteria': eligibility.get('eligibilityCriteria', ''),
            'min_age': eligibility.get('minimumAge', ''),
            'max_age': eligibility.get('maximumAge', ''),
            'gender': eligibility.get('gender', 'ALL'),
            'locations': locations,
            'contact': contact_info,
            'study_type': protocol_section.get('designModule', {}).get('studyType', ''),
            'last_update': status.get('lastUpdateDate', '')
        }
    
    def calculate_match_score(self, trial: Dict, patient_data: Dict, 
                            prediction_result: Dict) -> tuple[float, List[str]]:
        """Calculate match score and reasons for a trial-patient pair"""
        
        score = 0.0
        reasons = []
        
        # Base score for condition match
        prediction = prediction_result.get('prediction', '').lower()
        if 'aneurysm' in prediction:
            for condition in trial.get('conditions', []):
                if any(term in condition.lower() for term in ['aneurysm', 'hemorrhage', 'stroke']):
                    score += 40.0
                    reasons.append(f"Condition match: {condition}")
                    break
        
        # Age compatibility
        patient_age = patient_data.get('age')
        if patient_age:
            min_age_str = trial.get('min_age', '')
            max_age_str = trial.get('max_age', '')
            
            # Parse age strings (e.g., "18 Years", "65 Years")
            min_age = self._parse_age(min_age_str)
            max_age = self._parse_age(max_age_str)
            
            if min_age and max_age:
                if min_age <= patient_age <= max_age:
                    score += 20.0
                    reasons.append(f"Age compatible: {patient_age} within {min_age}-{max_age}")
                else:
                    score -= 10.0
                    reasons.append(f"Age incompatible: {patient_age} outside {min_age}-{max_age}")
            elif min_age and patient_age >= min_age:
                score += 15.0
                reasons.append(f"Minimum age met: {patient_age} >= {min_age}")
            elif max_age and patient_age <= max_age:
                score += 15.0
                reasons.append(f"Maximum age met: {patient_age} <= {max_age}")
        
        # Gender compatibility
        patient_gender = patient_data.get('sex', '').upper()
        trial_gender = trial.get('gender', 'ALL').upper()
        if trial_gender == 'ALL' or trial_gender == patient_gender:
            score += 10.0
            reasons.append("Gender compatible")
        else:
            score -= 15.0
            reasons.append(f"Gender incompatible: Trial for {trial_gender}, patient is {patient_gender}")
        
        # Location proximity (simplified - could be enhanced with actual distance calculation)
        # For now, just check if there are locations in the same country
        has_local_location = any(loc.get('country', '').lower() == 'united states' 
                               for loc in trial.get('locations', []))
        if has_local_location:
            score += 15.0
            reasons.append("Trial available in United States")
        
        # Study phase preference (Phase 2 and 3 generally preferred)
        phase = trial.get('phase', '').lower()
        if 'phase 2' in phase or 'phase 3' in phase:
            score += 10.0
            reasons.append(f"Favorable study phase: {phase}")
        elif 'phase 1' in phase:
            score += 5.0
            reasons.append(f"Early phase study: {phase}")
        
        # Confidence score bonus
        try:
            confidence_str = prediction_result.get('confidence', '0%')
            confidence = float(confidence_str.replace('%', ''))
            if confidence > 85:
                score += 5.0
                reasons.append(f"High prediction confidence: {confidence}%")
        except:
            pass
        
        return min(score, 100.0), reasons
    
    def _parse_age(self, age_str: str) -> Optional[int]:
        """Parse age string to integer"""
        if not age_str:
            return None
        
        # Extract number from strings like "18 Years", "65 Years"
        match = re.search(r'(\d+)', age_str)
        if match:
            return int(match.group(1))
        return None
    
    def find_matching_trials(self, prediction_result: Dict, patient_data: Dict, 
                           max_results: int = 5) -> List[ClinicalTrialMatch]:
        """Find and rank matching clinical trials for a patient"""
        
        # Determine search terms based on prediction
        prediction = prediction_result.get('prediction', '').lower()
        search_terms = []
        
        if 'aneurysm' in prediction:
            search_terms.extend(['aneurysm', 'cerebral aneurysm', 'intracranial aneurysm'])
        
        # Add related conditions
        search_terms.extend(['subarachnoid hemorrhage', 'stroke prevention'])
        
        all_trials = []
        
        # Search for trials with each term
        for term in search_terms[:3]:  # Limit to avoid too many API calls
            print(f"Searching for: {term}")
            trials = self.search_trials(
                condition=term,
                age=patient_data.get('age'),
                max_results=10
            )
            
            print(f"Found {len(trials)} trials for term: {term}")
            
            for trial_data in trials:
                trial_details = self.extract_trial_details(trial_data)
                if trial_details['nct_id']:  # Only process if we have valid data
                    all_trials.append(trial_details)
        
        # Remove duplicates based on NCT ID
        unique_trials = {trial['nct_id']: trial for trial in all_trials}.values()
        print(f"Total unique trials found: {len(list(unique_trials))}")
        
        # Calculate match scores
        matched_trials = []
        for trial in unique_trials:
            score, reasons = self.calculate_match_score(trial, patient_data, prediction_result)
            
            if score > 30:  # Only include trials with reasonable match scores
                # Format location string
                location_str = "Multiple locations"
                if trial['locations']:
                    first_loc = trial['locations'][0]
                    location_str = f"{first_loc['city']}, {first_loc['state']}"
                
                # Format contact info
                contact = trial['contact']
                contact_str = f"{contact.get('name', 'N/A')}"
                if contact.get('phone'):
                    contact_str += f" | {contact['phone']}"
                if contact.get('email'):
                    contact_str += f" | {contact['email']}"
                
                match = ClinicalTrialMatch(
                    nct_id=trial['nct_id'],
                    title=trial['title'],
                    status=trial['status'],
                    phase=trial['phase'],
                    condition=', '.join(trial['conditions'][:2]),  # First 2 conditions
                    location=location_str,
                    eligibility_criteria=trial['eligibility_criteria'][:200] + "..." if len(trial['eligibility_criteria']) > 200 else trial['eligibility_criteria'],
                    contact_info=contact_str,
                    match_score=score,
                    match_reasons=reasons
                )
                matched_trials.append(match)
        
        # Sort by match score and return top results
        matched_trials.sort(key=lambda x: x.match_score, reverse=True)
        return matched_trials[:max_results]
    
    def test_api_connection(self):
        """Test the API connection with a simple query"""
        print("Testing API connection...")
        
        # Try the simplest possible query
        test_params = {
            'format': 'json',
            'pageSize': 1
        }
        
        try:
            response = requests.get(self.base_url, params=test_params, headers=self.headers, timeout=10)
            print(f"Test URL: {response.url}")
            print(f"Test Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print("✅ API connection successful!")
                print(f"Total studies available: {data.get('totalCount', 'Unknown')}")
                return True
            else:
                print(f"❌ API test failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ API connection failed: {e}")
            return False
    
    def generate_trial_report(self, matched_trials: List[ClinicalTrialMatch], 
                            patient_data: Dict) -> str:
        """Generate a comprehensive clinical trial matching report"""
        
        if not matched_trials:
            return "🔍 No matching clinical trials found for this patient."
        
        report = "🧬 CLINICAL TRIAL MATCHING REPORT\n"
        report += "=" * 60 + "\n\n"
        
        # Patient summary
        report += f"👤 Patient Profile:\n"
        report += f"   Age: {patient_data.get('age', 'Unknown')}\n"
        report += f"   Gender: {patient_data.get('sex', 'Unknown')}\n"
        report += f"   Patient ID: {patient_data.get('patient_id', 'Unknown')}\n\n"
        
        report += f"📊 Found {len(matched_trials)} potentially matching trials:\n\n"
        
        for i, trial in enumerate(matched_trials, 1):
            report += f"🔬 TRIAL #{i} (Match Score: {trial.match_score:.1f}%)\n"
            report += f"   NCT ID: {trial.nct_id}\n"
            report += f"   Title: {trial.title}\n"
            report += f"   Status: {trial.status}\n"
            report += f"   Phase: {trial.phase}\n"
            report += f"   Condition: {trial.condition}\n"
            report += f"   Location: {trial.location}\n"
            report += f"   Contact: {trial.contact_info}\n"
            
            if trial.match_reasons:
                report += f"   ✅ Match Reasons:\n"
                for reason in trial.match_reasons[:3]:  # Top 3 reasons
                    report += f"      • {reason}\n"
            
            report += f"   🔗 More info: https://clinicaltrials.gov/study/{trial.nct_id}\n\n"
        
        report += "⚠️  IMPORTANT DISCLAIMER:\n"
        report += "This is an automated matching system. Please consult with the patient's\n"
        report += "healthcare provider before considering any clinical trial participation.\n"
        report += "All trial eligibility must be verified by qualified medical professionals.\n"
        
        return report
    
    def get_trial_summary(self, matched_trials: List[ClinicalTrialMatch]) -> Dict[str, Any]:
        """Get a structured summary of matched trials for integration"""
        
        if not matched_trials:
            return {
                'total_matches': 0,
                'top_match': None,
                'summary': 'No matching clinical trials found'
            }
        
        top_match = matched_trials[0]
        
        return {
            'total_matches': len(matched_trials),
            'top_match': {
                'nct_id': top_match.nct_id,
                'title': top_match.title,
                'match_score': top_match.match_score,
                'status': top_match.status,
                'phase': top_match.phase,
                'primary_reason': top_match.match_reasons[0] if top_match.match_reasons else 'Unknown'
            },
            'trials': [
                {
                    'nct_id': trial.nct_id,
                    'title': trial.title,
                    'match_score': trial.match_score,
                    'status': trial.status,
                    'phase': trial.phase,
                    'location': trial.location,
                    'contact': trial.contact_info,
                    'clinicaltrials_url': f"https://clinicaltrials.gov/study/{trial.nct_id}"
                }
                for trial in matched_trials
            ],
            'summary': f"Found {len(matched_trials)} matching trials. Top match: {top_match.title} ({top_match.match_score:.1f}% match)"
        }


# Example usage and testing
if __name__ == "__main__":
    # Create the agent
    agent = ClinicalTrialMatchingAgent()
    
    # Test API connection first
    if agent.test_api_connection():
        
        # Sample patient data
        patient_data = {
            'patient_id': 'P001',
            'age': 45,
            'sex': 'Female'
        }
        
        # Sample prediction result
        prediction_result = {
            'prediction': 'Brain aneurysm detected',
            'confidence': '92%'
        }
        
        # Find matching trials
        print("\n" + "="*60)
        print("SEARCHING FOR MATCHING CLINICAL TRIALS...")
        print("="*60)
        
        matched_trials = agent.find_matching_trials(prediction_result, patient_data)
        
        # Generate report
        report = agent.generate_trial_report(matched_trials, patient_data)
        print(report)
        
        # Get summary
        summary = agent.get_trial_summary(matched_trials)
        print("\n" + "="*60)
        print("SUMMARY:")
        print(json.dumps(summary, indent=2))
    
    else:
        print("❌ Cannot proceed without API connection")