import nibabel as nib
import pydicom
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from pathlib import Path


class ScanIntakeAgent:
    """
    Agent responsible for extracting and processing metadata from medical imaging scans.
    Handles NIfTI files and DICOM metadata extraction.
    """
    
    def __init__(self):
        self.supported_formats = ['.nii', '.nii.gz', '.dcm']
        
    def extract_nifti_metadata(self, nii_path: str) -> Dict[str, Any]:
        """Extract metadata from NIfTI files"""
        try:
            nii_img = nib.load(nii_path)
            header = nii_img.header
            
            # Basic image properties
            metadata = {
                'file_format': 'NIfTI',
                'file_path': nii_path,
                'file_size_mb': round(os.path.getsize(nii_path) / (1024 * 1024), 2),
                'dimensions': list(header.get_data_shape()),
                'voxel_size': list(header.get_zooms()),
                'data_type': str(header.get_data_dtype()),
                'orientation': str(nib.aff2axcodes(nii_img.affine)),
                'units': {
                    'spatial': header.get_xyzt_units()[0],
                    'temporal': header.get_xyzt_units()[1]
                }
            }
            
            # Additional NIfTI specific metadata
            if hasattr(header, 'get_intent'):
                metadata['intent'] = header.get_intent()
            
            # Extract any available scan parameters
            if hasattr(header, 'get_slope_inter'):
                slope, inter = header.get_slope_inter()
                metadata['intensity_scaling'] = {'slope': slope, 'intercept': inter}
                
            return metadata
            
        except Exception as e:
            return {'error': f'Failed to extract NIfTI metadata: {str(e)}'}
    
    def extract_dicom_metadata(self, dicom_path: str) -> Dict[str, Any]:
        """Extract metadata from DICOM files"""
        try:
            ds = pydicom.dcmread(dicom_path)
            
            metadata = {
                'file_format': 'DICOM',
                'file_path': dicom_path,
                'file_size_mb': round(os.path.getsize(dicom_path) / (1024 * 1024), 2),
                'patient_info': {
                    'patient_id': getattr(ds, 'PatientID', 'Unknown'),
                    'patient_name': str(getattr(ds, 'PatientName', 'Unknown')),
                    'patient_age': getattr(ds, 'PatientAge', 'Unknown'),
                    'patient_sex': getattr(ds, 'PatientSex', 'Unknown'),
                    'patient_birth_date': getattr(ds, 'PatientBirthDate', 'Unknown')
                },
                'study_info': {
                    'study_date': getattr(ds, 'StudyDate', 'Unknown'),
                    'study_time': getattr(ds, 'StudyTime', 'Unknown'),
                    'study_description': getattr(ds, 'StudyDescription', 'Unknown'),
                    'series_description': getattr(ds, 'SeriesDescription', 'Unknown'),
                    'modality': getattr(ds, 'Modality', 'Unknown')
                },
                'acquisition_info': {
                    'manufacturer': getattr(ds, 'Manufacturer', 'Unknown'),
                    'manufacturer_model': getattr(ds, 'ManufacturerModelName', 'Unknown'),
                    'software_version': getattr(ds, 'SoftwareVersions', 'Unknown'),
                    'slice_thickness': getattr(ds, 'SliceThickness', 'Unknown'),
                    'pixel_spacing': getattr(ds, 'PixelSpacing', 'Unknown'),
                    'image_orientation': getattr(ds, 'ImageOrientationPatient', 'Unknown')
                }
            }
            
            # Add scan-specific parameters if available
            scan_params = {}
            if hasattr(ds, 'RepetitionTime'):
                scan_params['repetition_time'] = ds.RepetitionTime
            if hasattr(ds, 'EchoTime'):
                scan_params['echo_time'] = ds.EchoTime
            if hasattr(ds, 'FlipAngle'):
                scan_params['flip_angle'] = ds.FlipAngle
            if hasattr(ds, 'MagneticFieldStrength'):
                scan_params['magnetic_field_strength'] = ds.MagneticFieldStrength
                
            if scan_params:
                metadata['scan_parameters'] = scan_params
                
            return metadata
            
        except Exception as e:
            return {'error': f'Failed to extract DICOM metadata: {str(e)}'}
    
    def validate_scan_quality(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate scan quality based on metadata"""
        quality_report = {
            'overall_quality': 'Good',
            'issues': [],
            'recommendations': []
        }
        
        # Check file size
        if 'file_size_mb' in metadata:
            if metadata['file_size_mb'] < 1:
                quality_report['issues'].append('File size unusually small')
                quality_report['overall_quality'] = 'Warning'
            elif metadata['file_size_mb'] > 500:
                quality_report['issues'].append('File size unusually large')
        
        # Check dimensions for NIfTI
        if 'dimensions' in metadata:
            dims = metadata['dimensions']
            if len(dims) >= 3:
                if any(d < 32 for d in dims[:3]):
                    quality_report['issues'].append('Low spatial resolution detected')
                    quality_report['overall_quality'] = 'Warning'
                if any(d > 1024 for d in dims[:3]):
                    quality_report['recommendations'].append('Consider downsampling for faster processing')
        
        # Check voxel size
        if 'voxel_size' in metadata:
            voxel_sizes = metadata['voxel_size']
            if len(voxel_sizes) >= 3:
                if any(v > 3.0 for v in voxel_sizes[:3]):
                    quality_report['issues'].append('Large voxel size may affect analysis accuracy')
                    quality_report['overall_quality'] = 'Warning'
        
        return quality_report
    
    def process_scan(self, scan_path: str, patient_id: Optional[int] = None) -> Dict[str, Any]:
        """Main method to process a scan and extract all relevant metadata"""
        
        if not os.path.exists(scan_path):
            return {'error': f'Scan file not found: {scan_path}'}
        
        file_ext = Path(scan_path).suffix.lower()
        
        # Extract metadata based on file type
        if file_ext in ['.nii', '.gz']:
            metadata = self.extract_nifti_metadata(scan_path)
        elif file_ext == '.dcm':
            metadata = self.extract_dicom_metadata(scan_path)
        else:
            return {'error': f'Unsupported file format: {file_ext}'}
        
        if 'error' in metadata:
            return metadata
        
        # Add processing timestamp
        metadata['processing_timestamp'] = datetime.now().isoformat()
        
        # Add patient ID if provided
        if patient_id:
            metadata['patient_id'] = patient_id
        
        # Validate scan quality
        quality_report = self.validate_scan_quality(metadata)
        metadata['quality_assessment'] = quality_report
        
        # Add scan compatibility info
        metadata['ai_processing_ready'] = self._check_ai_compatibility(metadata)
        
        return metadata
    
    def _check_ai_compatibility(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Check if scan is compatible with AI processing pipeline"""
        compatibility = {
            'compatible': True,
            'issues': [],
            'preprocessing_required': []
        }
        
        # Check for common AI processing requirements
        if 'dimensions' in metadata:
            dims = metadata['dimensions']
            if len(dims) >= 3:
                # Check if dimensions are powers of 2 (often required for neural networks)
                if not all(d > 0 and (d & (d-1)) == 0 for d in dims[:3]):
                    compatibility['preprocessing_required'].append('Dimension padding to power of 2')
        
        # Check data type compatibility
        if 'data_type' in metadata:
            if metadata['data_type'] not in ['float32', 'float64', 'uint8', 'uint16']:
                compatibility['preprocessing_required'].append('Data type conversion required')
        
        # Check orientation
        if 'orientation' in metadata:
            if metadata['orientation'] != "('R', 'A', 'S')":  # RAS orientation
                compatibility['preprocessing_required'].append('Reorientation to RAS required')
        
        if compatibility['preprocessing_required']:
            compatibility['compatible'] = False
            
        return compatibility
    
    def generate_intake_report(self, metadata: Dict[str, Any]) -> str:
        """Generate a human-readable intake report"""
        if 'error' in metadata:
            return f"❌ Scan Intake Failed: {metadata['error']}"
        
        report = "🔍 SCAN INTAKE REPORT\n"
        report += "=" * 50 + "\n\n"
        
        # Basic info
        report += f"📁 File: {metadata.get('file_path', 'Unknown')}\n"
        report += f"📏 Size: {metadata.get('file_size_mb', 'Unknown')} MB\n"
        report += f"🗂️ Format: {metadata.get('file_format', 'Unknown')}\n"
        report += f"⏰ Processed: {metadata.get('processing_timestamp', 'Unknown')}\n\n"
        
        # Dimensions and technical specs
        if 'dimensions' in metadata:
            report += f"📐 Dimensions: {metadata['dimensions']}\n"
        if 'voxel_size' in metadata:
            report += f"🔬 Voxel Size: {metadata['voxel_size']}\n"
        if 'data_type' in metadata:
            report += f"💾 Data Type: {metadata['data_type']}\n\n"
        
        # Quality assessment
        if 'quality_assessment' in metadata:
            qa = metadata['quality_assessment']
            status_emoji = "✅" if qa['overall_quality'] == 'Good' else "⚠️"
            report += f"{status_emoji} Quality: {qa['overall_quality']}\n"
            if qa['issues']:
                report += "Issues:\n"
                for issue in qa['issues']:
                    report += f"  • {issue}\n"
        
        # AI compatibility
        if 'ai_processing_ready' in metadata:
            ai_compat = metadata['ai_processing_ready']
            compat_emoji = "✅" if ai_compat['compatible'] else "🔄"
            report += f"{compat_emoji} AI Ready: {'Yes' if ai_compat['compatible'] else 'Preprocessing Required'}\n"
            
            if ai_compat['preprocessing_required']:
                report += "Preprocessing needed:\n"
                for req in ai_compat['preprocessing_required']:
                    report += f"  • {req}\n"
        
        return report