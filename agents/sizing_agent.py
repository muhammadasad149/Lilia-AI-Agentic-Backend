import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

class EnhancedSizingAgent:
    def __init__(self, pixel_to_mm_ratio=0.39):
        self.pixel_to_mm_ratio = pixel_to_mm_ratio

    def load_image_from_base64(self, base64_string):
        """Load image from base64 string"""
        try:
            # Remove data:image/png;base64, prefix if present
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_string)
            
            # Convert bytes to PIL Image
            pil_image = Image.open(BytesIO(image_bytes))
            
            # Convert PIL to OpenCV format
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array and change RGB to BGR for OpenCV
            image = np.array(pil_image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image from base64: {str(e)}")

    def load_image_from_path(self, image_path):
        """Load image from file path (original method)"""
        if isinstance(image_path, str):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Image path must have a suitable image.")
            return image
        else:
            raise ValueError("Image path must be a string.")

    def create_mask_from_binary(self, binary_image):
        """Create mask from already binary/grayscale segmentation image"""
        # Convert to grayscale if needed
        if len(binary_image.shape) == 3:
            gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = binary_image
        
        # Threshold to ensure binary mask
        _, binary_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return binary_mask

    def create_mask_from_color(self, image):
        """Create mask by detecting red regions (original method)"""
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define red color range
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        # Combine masks for lower and upper red hues
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        return red_mask

    def extract_aneurysm_measurements(self, mask):
        """Extract aneurysm measurements from binary mask"""
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (main aneurysm)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get all bounding boxes for multiple detected regions
            bounding_boxes = []
            areas = []
            for contour in contours:
                if cv2.contourArea(contour) > 10:  # Filter out very small regions
                    x, y, w, h = cv2.boundingRect(contour)
                    area = cv2.contourArea(contour)
                    bounding_boxes.append((x, y, w, h))
                    areas.append(area)
            
            # Main measurements from largest contour
            main_bbox = cv2.boundingRect(largest_contour)
            main_area = cv2.contourArea(largest_contour)
            
            return {
                'main_bbox': main_bbox,
                'main_contour': largest_contour,
                'main_area': main_area,
                'all_bboxes': bounding_boxes,
                'all_areas': areas,
                'total_regions': len(bounding_boxes)
            }
        else:
            return None

    def calculate_measurements(self, extraction_results):
        """Calculate physical measurements from extraction results"""
        if extraction_results is None:
            return {
                "length_mm": 0,
                "width_mm": 0,
                "area_mm2": 0,
                "total_regions": 0,
                "measurements_per_region": [],
                "pixel_to_mm_ratio": self.pixel_to_mm_ratio
            }
        
        # Main aneurysm measurements
        x, y, w, h = extraction_results['main_bbox']
        length_px = max(w, h)
        width_px = min(w, h)
        area_px2 = extraction_results['main_area']
        
        # Convert to mm
        main_measurements = {
            "length_mm": round(length_px * self.pixel_to_mm_ratio, 2),
            "width_mm": round(width_px * self.pixel_to_mm_ratio, 2),
            "area_mm2": round(area_px2 * (self.pixel_to_mm_ratio ** 2), 2)
        }
        
        # Measurements for all detected regions
        region_measurements = []
        for i, (bbox, area_px) in enumerate(zip(extraction_results['all_bboxes'], extraction_results['all_areas'])):
            x, y, w, h = bbox
            region_length_px = max(w, h)
            region_width_px = min(w, h)
            
            region_measurements.append({
                "region_id": i + 1,
                "length_mm": round(region_length_px * self.pixel_to_mm_ratio, 2),
                "width_mm": round(region_width_px * self.pixel_to_mm_ratio, 2),
                "area_mm2": round(area_px * (self.pixel_to_mm_ratio ** 2), 2)
            })
        
        return {
            "length_mm": main_measurements["length_mm"],
            "width_mm": main_measurements["width_mm"],
            "area_mm2": main_measurements["area_mm2"],
            "total_regions": extraction_results['total_regions'],
            "measurements_per_region": region_measurements,
            "pixel_to_mm_ratio": self.pixel_to_mm_ratio
        }

    def measure_from_base64(self, base64_image, is_segmentation_mask=True):
        """Measure aneurysm from base64 image"""
        try:
            # Load image from base64
            image = self.load_image_from_base64(base64_image)
            
            # Create mask based on image type
            if is_segmentation_mask:
                # For segmentation results (binary/grayscale images)
                mask = self.create_mask_from_binary(image)
            else:
                # For colored images with red regions
                mask = self.create_mask_from_color(image)
            
            # Extract measurements
            extraction_results = self.extract_aneurysm_measurements(mask)
            measurements = self.calculate_measurements(extraction_results)
            
            return measurements
            
        except Exception as e:
            return {
                "error": f"Measurement failed: {str(e)}",
                "length_mm": 0,
                "width_mm": 0,
                "area_mm2": 0,
                "total_regions": 0,
                "measurements_per_region": [],
                "pixel_to_mm_ratio": self.pixel_to_mm_ratio
            }

    def measure_from_path(self, image_path, use_color_detection=True):
        """Measure aneurysm from file path (original method)"""
        try:
            image = self.load_image_from_path(image_path)
            
            if use_color_detection:
                mask = self.create_mask_from_color(image)
            else:
                mask = self.create_mask_from_binary(image)
            
            extraction_results = self.extract_aneurysm_measurements(mask)
            measurements = self.calculate_measurements(extraction_results)
            
            return measurements
            
        except Exception as e:
            return {
                "error": f"Measurement failed: {str(e)}",
                "length_mm": 0,
                "width_mm": 0,
                "area_mm2": 0,
                "total_regions": 0,
                "measurements_per_region": [],
                "pixel_to_mm_ratio": self.pixel_to_mm_ratio
            }

    def analyze_segmentation_results(self, segmentation_results):
        """Analyze all segmentation images and return comprehensive measurements"""
        if not segmentation_results or 'region_images' not in segmentation_results:
            return {
                "error": "No segmentation results provided",
                "slice_measurements": [],
                "summary": {
                    "max_length_mm": 0,
                    "max_width_mm": 0,
                    "max_area_mm2": 0,
                    "total_detected_regions": 0,
                    "slices_with_detections": 0
                }
            }
        
        slice_measurements = []
        all_lengths = []
        all_widths = []
        all_areas = []
        total_regions = 0
        slices_with_detections = 0
        
        # Analyze each slice
        region_images = segmentation_results.get('region_images', [])
        slice_indices = segmentation_results.get('slice_indices', [])
        
        for i, base64_image in enumerate(region_images):
            slice_index = slice_indices[i] if i < len(slice_indices) else i
            
            # Measure this slice
            measurements = self.measure_from_base64(base64_image, is_segmentation_mask=True)
            
            # Add slice information
            measurements['slice_index'] = slice_index
            measurements['slice_number'] = i + 1
            
            slice_measurements.append(measurements)
            
            # Collect statistics
            if measurements.get('total_regions', 0) > 0:
                slices_with_detections += 1
                total_regions += measurements.get('total_regions', 0)
                
                if measurements.get('length_mm', 0) > 0:
                    all_lengths.append(measurements['length_mm'])
                if measurements.get('width_mm', 0) > 0:
                    all_widths.append(measurements['width_mm'])
                if measurements.get('area_mm2', 0) > 0:
                    all_areas.append(measurements['area_mm2'])
        
        # Calculate summary statistics
        summary = {
            "max_length_mm": max(all_lengths) if all_lengths else 0,
            "max_width_mm": max(all_widths) if all_widths else 0,
            "max_area_mm2": max(all_areas) if all_areas else 0,
            "avg_length_mm": round(np.mean(all_lengths), 2) if all_lengths else 0,
            "avg_width_mm": round(np.mean(all_widths), 2) if all_widths else 0,
            "avg_area_mm2": round(np.mean(all_areas), 2) if all_areas else 0,
            "total_detected_regions": total_regions,
            "slices_with_detections": slices_with_detections,
            "total_slices_analyzed": len(region_images),
            "pixel_to_mm_ratio": self.pixel_to_mm_ratio
        }
        
        return {
            "slice_measurements": slice_measurements,
            "summary": summary
        }


# # Example usage
# if __name__ == "__main__":
#     # Initialize the enhanced sizing agent
#     sizing_agent = EnhancedSizingAgent(pixel_to_mm_ratio=0.39)
    
#     # Example with base64 image
#     # base64_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
#     # measurements = sizing_agent.measure_from_base64(base64_image, is_segmentation_mask=True)
#     # print("Measurements:", measurements)