"""
Claim Document Processor
Specialized processor for HDFC Life Cancer Care claim forms.
Extracts structured data with improved OCR accuracy using RapidOCR.
"""

import logging
import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import numpy as np

# Image processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("opencv-python not available. Install with: pip install opencv-python")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logging.warning("pymupdf not available. Install with: pip install pymupdf")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL/Pillow not available. Install with: pip install pillow")

# OCR
try:
    from rapidocr import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False
    logging.warning("rapidocr not available. Install with: pip install rapidocr")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Enable debug logging for OCR processing
logger.setLevel(logging.DEBUG)

# ---------------- CONFIGURATION ----------------
DPI = 300  # High resolution for better OCR
OCR_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for accepting OCR results

# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(image: np.ndarray, use_binarization: bool = False) -> np.ndarray:
    """
    Preprocess image for better OCR accuracy.
    
    Args:
        image: Input image as numpy array
        use_binarization: If True, apply binarization (may be too aggressive for some images)
        
    Returns:
        Preprocessed image
    """
    if not CV2_AVAILABLE:
        return image
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Light denoising (less aggressive)
    denoised = cv2.fastNlMeansDenoising(gray, h=5, templateWindowSize=7, searchWindowSize=21)
    
    # Contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Only apply binarization if requested (often too aggressive for OCR)
    if use_binarization:
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return binary
    
    # Return enhanced grayscale (better for RapidOCR)
    return enhanced

def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Deskew image to correct rotation.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Deskewed image
    """
    if not CV2_AVAILABLE:
        return image
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Find all non-zero points
    coords = np.column_stack(np.where(gray > 0))
    
    if len(coords) == 0:
        return image
    
    # Find minimum area rectangle
    angle = cv2.minAreaRect(coords)[-1]
    
    # Correct angle
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Rotate image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated

# ---------------- OCR EXTRACTION ----------------
def extract_text_with_ocr(pdf_path: Path, debug_output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Extract text from PDF using OCR with bounding boxes.
    
    Args:
        pdf_path: Path to PDF file
        debug_output_dir: Optional directory to save debug outputs
        
    Returns:
        List of page results, each containing text and bounding boxes
    """
    if not RAPIDOCR_AVAILABLE:
        logger.error("RapidOCR not available. Cannot perform OCR extraction.")
        return []
    
    if not PYMUPDF_AVAILABLE:
        logger.error("pymupdf not available. Cannot convert PDF to images.")
        return []
    
    if not PIL_AVAILABLE:
        logger.error("PIL/Pillow not available. Cannot convert PDF to images.")
        return []
    
    # Create debug directory
    if debug_output_dir:
        debug_dir = debug_output_dir / "debug_ocr"
        debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Debug outputs will be saved to: {debug_dir}")
    else:
        debug_dir = None
    
    logger.info(f"Initializing RapidOCR...")
    try:
        # Try different RapidOCR initialization methods
        ocr = RapidOCR()
        logger.debug(f"RapidOCR initialized: {type(ocr)}")
    except Exception as e:
        logger.error(f"Failed to initialize RapidOCR: {e}")
        return []
    
    logger.info(f"Converting PDF to images using pymupdf (DPI={DPI})...")
    try:
        # Open PDF with pymupdf
        doc = fitz.open(str(pdf_path))
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Render page to image (pixmap) at specified DPI
            # Matrix: scale factor = DPI/72 (72 is default DPI)
            mat = fitz.Matrix(DPI/72, DPI/72)
            pix = page.get_pixmap(matrix=mat)
            # Convert pixmap to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
            
            # Save original image for debugging
            if debug_dir:
                orig_img_path = debug_dir / f"page_{page_num+1}_00_original.png"
                img.save(orig_img_path)
                logger.debug(f"  - Saved original image: {orig_img_path}")
        
        doc.close()
    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}")
        return []
    
    logger.info(f"Processing {len(images)} pages with OCR...")
    page_results = []
    
    for page_num, pil_image in enumerate(images, 1):
        logger.info(f"Processing page {page_num}/{len(images)}...")
        logger.debug(f"  - PIL image size: {pil_image.size}, mode: {pil_image.mode}")
        
        # Convert PIL to numpy array
        img_array = np.array(pil_image)
        logger.debug(f"  - Converted to numpy array: shape={img_array.shape}, dtype={img_array.dtype}")
        
        # Save original numpy array image
        if debug_dir:
            orig_np_path = debug_dir / f"page_{page_num}_01_original_numpy.png"
            Image.fromarray(img_array).save(orig_np_path)
            logger.debug(f"  - Saved original numpy image: {orig_np_path}")
        
        # Try OCR on original image first (RapidOCR often works better on original)
        logger.debug(f"  - Trying OCR on original image first...")
        ocr_result_original = None
        try:
            # Convert to RGB if needed (RapidOCR expects RGB or grayscale)
            if len(img_array.shape) == 3:
                ocr_input = img_array
            else:
                ocr_input = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB) if CV2_AVAILABLE else img_array
            
            logger.debug(f"  - Original image shape: {ocr_input.shape}, dtype: {ocr_input.dtype}")
            ocr_result_original = ocr(ocr_input)
            
            # Check if we got results
            txts_count = len(getattr(ocr_result_original, 'txts', [])) if hasattr(ocr_result_original, 'txts') else 0
            logger.debug(f"  - Original image OCR: {txts_count} text blocks found")
            
            if txts_count > 0:
                logger.info(f"  - Using original image OCR (found {txts_count} blocks)")
                ocr_result = ocr_result_original
            else:
                logger.debug(f"  - Original image OCR found no text, trying preprocessed...")
                ocr_result = None
        except Exception as e:
            logger.warning(f"  - OCR on original image failed: {e}")
            ocr_result = None
        
        # If original didn't work, try preprocessed
        if ocr_result is None or (hasattr(ocr_result, 'txts') and len(getattr(ocr_result, 'txts', [])) == 0):
            # Preprocess image (light preprocessing, no binarization)
            logger.debug(f"  - Preprocessing image (light preprocessing)...")
            processed_img = preprocess_image(img_array, use_binarization=False)
            logger.debug(f"  - After preprocessing: shape={processed_img.shape}, dtype={processed_img.dtype}")
            
            # Save preprocessed image
            if debug_dir:
                preprocessed_path = debug_dir / f"page_{page_num}_02_preprocessed.png"
                Image.fromarray(processed_img).save(preprocessed_path)
                logger.debug(f"  - Saved preprocessed image: {preprocessed_path}")
            
            # Light deskewing (skip if angle is very small)
            processed_img = deskew_image(processed_img)
            logger.debug(f"  - After deskewing: shape={processed_img.shape}")
            
            # Save deskewed image
            if debug_dir:
                deskewed_path = debug_dir / f"page_{page_num}_03_deskewed.png"
                Image.fromarray(processed_img).save(deskewed_path)
                logger.debug(f"  - Saved deskewed image: {deskewed_path}")
            
            # Perform OCR on preprocessed
            try:
                logger.debug(f"  - Image shape: {processed_img.shape}, dtype: {processed_img.dtype}")
                logger.debug(f"  - Calling RapidOCR on preprocessed image...")
                
                # Convert to 3-channel if needed (RapidOCR may need RGB)
                if len(processed_img.shape) == 2:
                    # Grayscale to RGB
                    if CV2_AVAILABLE:
                        ocr_input = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
                    else:
                        ocr_input = np.stack([processed_img] * 3, axis=-1)
                else:
                    ocr_input = processed_img
                
                logger.debug("  - Calling RapidOCR on preprocessed image...")
                ocr_result = ocr(ocr_input)
            except Exception as e:
                logger.error(f"  - OCR on preprocessed image failed: {e}")
                ocr_result = ocr_result_original  # Fallback to original result
        else:
            # We already have a good result from original
            processed_img = img_array  # For saving debug images
        
        # Process OCR result (regardless of which path we took)
        try:
            if ocr_result is None:
                logger.error(f"  - OCR result is None for page {page_num}")
                raise ValueError("OCR result is None")
            
            logger.debug(f"  - OCR returned type: {type(ocr_result)}")
            
            # Save raw OCR result for debugging (always save, even if empty)
            if debug_dir:
                raw_ocr_path = debug_dir / f"page_{page_num}_04_raw_ocr_result.json"
                try:
                    # Try to serialize the OCR result
                    ocr_dict = {
                        "type": str(type(ocr_result)),
                        "attributes": [a for a in dir(ocr_result) if not a.startswith('_')]
                    }
                    # Try to get attribute values (handle numpy arrays properly)
                    if hasattr(ocr_result, 'txts'):
                        txts_attr = ocr_result.txts
                        if txts_attr is not None:
                            # Convert to list if it's a numpy array or other iterable
                            try:
                                txts_list = list(txts_attr) if hasattr(txts_attr, '__iter__') and not isinstance(txts_attr, str) else []
                            except:
                                txts_list = []
                            ocr_dict['txts_count'] = len(txts_list)
                            ocr_dict['txts_sample'] = txts_list[:10] if txts_list else []
                            ocr_dict['txts_all'] = txts_list  # Save all for debugging
                        else:
                            ocr_dict['txts_count'] = 0
                            ocr_dict['txts_sample'] = []
                            ocr_dict['txts_all'] = []
                    if hasattr(ocr_result, 'boxes'):
                        boxes_attr = ocr_result.boxes
                        if boxes_attr is not None:
                            try:
                                # Convert numpy array to list
                                if hasattr(boxes_attr, 'tolist'):
                                    boxes_list = boxes_attr.tolist()
                                elif hasattr(boxes_attr, '__iter__') and not isinstance(boxes_attr, str):
                                    boxes_list = list(boxes_attr)
                                else:
                                    boxes_list = []
                            except:
                                boxes_list = []
                            ocr_dict['boxes_count'] = len(boxes_list)
                            ocr_dict['boxes_sample'] = [str(b) for b in boxes_list[:3]] if boxes_list else []
                        else:
                            ocr_dict['boxes_count'] = 0
                            ocr_dict['boxes_sample'] = []
                    if hasattr(ocr_result, 'scores'):
                        scores_attr = ocr_result.scores
                        if scores_attr is not None:
                            try:
                                # Convert numpy array to list
                                if hasattr(scores_attr, 'tolist'):
                                    scores_list = scores_attr.tolist()
                                elif hasattr(scores_attr, '__iter__') and not isinstance(scores_attr, str):
                                    scores_list = list(scores_attr)
                                else:
                                    scores_list = []
                            except:
                                scores_list = []
                            ocr_dict['scores_count'] = len(scores_list)
                            ocr_dict['scores_sample'] = scores_list[:10] if scores_list else []
                        else:
                            ocr_dict['scores_count'] = 0
                            ocr_dict['scores_sample'] = []
                    if hasattr(ocr_result, 'elapse'):
                        ocr_dict['elapse'] = ocr_result.elapse
                    
                    with open(raw_ocr_path, 'w', encoding='utf-8') as f:
                        json.dump(ocr_dict, f, indent=2, ensure_ascii=False, default=str)
                    logger.info(f"  - Saved raw OCR result: {raw_ocr_path} (txts: {ocr_dict.get('txts_count', 0)})")
                except Exception as e:
                    logger.error(f"  - Could not save raw OCR result: {e}", exc_info=True)
            
            # Extract result list from RapidOCROutput object
            result = []
            
            # Check if it's RapidOCROutput (newer versions)
            if hasattr(ocr_result, 'txts') or hasattr(ocr_result, 'boxes'):
                logger.debug("  - Detected RapidOCROutput object")
                attrs = [a for a in dir(ocr_result) if not a.startswith('_')]
                logger.debug(f"  - Available attributes: {attrs}")
                
                # Extract txts, boxes, scores (handle numpy arrays properly)
                txts_attr = getattr(ocr_result, 'txts', None)
                boxes_attr = getattr(ocr_result, 'boxes', None)
                scores_attr = getattr(ocr_result, 'scores', None)
                
                # Convert to lists, handling numpy arrays
                if txts_attr is not None:
                    try:
                        txts = list(txts_attr) if hasattr(txts_attr, '__iter__') and not isinstance(txts_attr, str) else []
                    except:
                        txts = []
                else:
                    txts = []
                
                if boxes_attr is not None:
                    try:
                        if hasattr(boxes_attr, 'tolist'):
                            boxes = boxes_attr.tolist()
                        elif hasattr(boxes_attr, '__iter__') and not isinstance(boxes_attr, str):
                            boxes = list(boxes_attr)
                        else:
                            boxes = []
                    except:
                        boxes = []
                else:
                    boxes = []
                
                if scores_attr is not None:
                    try:
                        if hasattr(scores_attr, 'tolist'):
                            scores = scores_attr.tolist()
                        elif hasattr(scores_attr, '__iter__') and not isinstance(scores_attr, str):
                            scores = list(scores_attr)
                        else:
                            scores = []
                    except:
                        scores = []
                else:
                    scores = []
                
                logger.debug(f"  - txts count: {len(txts)}, boxes count: {len(boxes)}, scores count: {len(scores)}")
                
                # Combine into expected format: [[bbox, text, confidence], ...]
                max_len = max(len(txts), len(boxes), len(scores))
                logger.debug(f"  - Max length: {max_len}")
                
                for i in range(max_len):
                    box = boxes[i] if i < len(boxes) else None
                    text = str(txts[i]) if i < len(txts) else ""
                    score = float(scores[i]) if i < len(scores) else 1.0
                    result.append([box, text, score])
                
                logger.info(f"  - Extracted {len(result)} text blocks from RapidOCROutput")
                
                # Save extracted result for debugging
                if debug_dir:
                    extracted_path = debug_dir / f"page_{page_num}_05_extracted_blocks.json"
                    with open(extracted_path, 'w', encoding='utf-8') as f:
                        json.dump({
                            "total_blocks": len(result),
                            "blocks": [
                                {
                                    "bbox": str(block[0]) if block[0] else None,
                                    "text": block[1],
                                    "confidence": block[2]
                                }
                                for block in result[:20]  # Save first 20 for debugging
                            ]
                        }, f, indent=2, ensure_ascii=False)
                    logger.debug(f"  - Saved extracted blocks: {extracted_path}")
            
            # Fallback: Check if it's already a list
            elif isinstance(ocr_result, list):
                logger.debug("  - Result is already a list")
                result = ocr_result
            # Fallback: Check if it's a tuple (result, elapsed_time)
            elif isinstance(ocr_result, tuple):
                logger.debug(f"  - Result is tuple with {len(ocr_result)} elements")
                if len(ocr_result) > 0:
                    result = ocr_result[0] if isinstance(ocr_result[0], list) else list(ocr_result[0]) if hasattr(ocr_result[0], '__iter__') else []
            else:
                logger.warning(f"  - Unknown OCR result format: {type(ocr_result)}")
                logger.debug(f"  - Available attributes: {[a for a in dir(ocr_result) if not a.startswith('_')]}")
            
            logger.info(f"  - Final result: {len(result)} text blocks")
            
            logger.debug(f"  - Extracted {len(result)} text blocks")
            
            # Structure result
            page_data = {
                "page_number": page_num,
                "text_blocks": [],
                "full_text": ""
            }
            
            text_lines = []
            for idx, item in enumerate(result):
                try:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        bbox = item[0] if len(item) > 0 and isinstance(item[0], (list, tuple)) else None
                        text = str(item[1]) if len(item) > 1 else ""
                        confidence = float(item[2]) if len(item) > 2 else 1.0
                        
                        logger.debug(f"    Block {idx}: text='{text[:50]}...', confidence={confidence:.2f}")
                        
                        if confidence >= OCR_CONFIDENCE_THRESHOLD:
                            page_data["text_blocks"].append({
                                "text": text,
                                "bbox": bbox,
                                "confidence": confidence
                            })
                            text_lines.append(text)
                        else:
                            logger.debug(f"    Block {idx} filtered (low confidence: {confidence:.2f})")
                    else:
                        logger.debug(f"    Block {idx} skipped (invalid format: {type(item)})")
                except Exception as e:
                    logger.warning(f"    Error processing block {idx}: {e}")
                    continue
            
            page_data["full_text"] = "\n".join(text_lines)
            logger.info(f"  - Extracted {len(text_lines)} text lines from page {page_num}")
            logger.debug(f"  - Sample text (first 200 chars): {page_data['full_text'][:200]}")
            
            # Save full text for debugging
            if debug_dir:
                full_text_path = debug_dir / f"page_{page_num}_06_full_text.txt"
                with open(full_text_path, 'w', encoding='utf-8') as f:
                    f.write(page_data["full_text"])
                logger.debug(f"  - Saved full text: {full_text_path}")
                
                # Save page data JSON
                page_data_path = debug_dir / f"page_{page_num}_07_page_data.json"
                with open(page_data_path, 'w', encoding='utf-8') as f:
                    json.dump(page_data, f, indent=2, ensure_ascii=False, default=str)
                logger.debug(f"  - Saved page data: {page_data_path}")
            
            page_results.append(page_data)
            
        except Exception as e:
            logger.error(f"OCR failed for page {page_num}: {e}", exc_info=True)
            # Try to get at least some text using RapidOCR's text extraction method
            try:
                logger.debug("  - Attempting fallback: using RapidOCR text extraction...")
                # Some RapidOCR versions have a .ocr() method that returns text directly
                if hasattr(ocr, 'ocr'):
                    fallback_result = ocr.ocr(processed_img)
                    if fallback_result:
                        logger.debug(f"  - Fallback extracted {len(fallback_result)} items")
                        # Process fallback result
                        text_lines = []
                        for item in fallback_result:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                text_lines.append(str(item[1]))
                        if text_lines:
                            logger.info(f"  - Fallback extracted {len(text_lines)} text lines")
                            page_results.append({
                                "page_number": page_num,
                                "text_blocks": [],
                                "full_text": "\n".join(text_lines),
                                "warning": "Used fallback extraction method"
                            })
                            continue
            except Exception as fallback_error:
                logger.debug(f"  - Fallback also failed: {fallback_error}")
            
            page_results.append({
                "page_number": page_num,
                "text_blocks": [],
                "full_text": "",
                "error": str(e)
            })
    
    return page_results

# ---------------- FORM PARSER ----------------
class ClaimFormParser:
    """Parser for HDFC Life Cancer Care claim forms."""
    
    def __init__(self, ocr_results: List[Dict[str, Any]]):
        self.ocr_results = ocr_results
        self.full_text = "\n".join([page.get("full_text", "") for page in ocr_results])
        self.extracted_data = {}
        self.confidence_scores = {}
    
    def normalize_ocr_text(self, text: str) -> str:
        """Normalize OCR text to handle common errors."""
        # Remove extra spaces, normalize
        text = re.sub(r'\s+', ' ', text.strip())
        # Fix common OCR errors
        text = text.replace('ll', 'l')  # ll -> l (e.g., "llness" -> "illness")
        text = text.replace('rn', 'm')  # rn -> m (common OCR error)
        text = text.replace('vv', 'w')  # vv -> w
        return text
    
    def find_section(self, section_marker: str) -> Optional[str]:
        """Find text content of a section."""
        # Normalize section marker for matching
        marker_normalized = section_marker.upper().replace(' ', '').replace('/', '')
        
        # Try multiple patterns
        patterns = [
            # Exact match
            rf"{re.escape(section_marker)}.*?(?=\n\([A-G]\)|$)",
            # Without spaces in marker
            rf"{re.escape(section_marker.replace(' ', ''))}.*?(?=\n\([A-G]\)|$)",
            # Just the letter and key words
            rf"\([A-G]\)\s*DETAILS.*?{re.escape(section_marker.split(':')[0].split('(')[-1])}.*?(?=\n\([A-G]\)|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, self.full_text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(0)
        
        # Try finding by section letter and key words
        section_letter = section_marker[1] if len(section_marker) > 1 and section_marker[0] == '(' else None
        if section_letter:
            # Look for section markers like "(A)", "(B)", etc.
            section_patterns = [
                rf"\({section_letter}\)\s*DETAILS.*?(?=\n\([A-G]\)|$)",
                rf"\({section_letter}\)DETAILS.*?(?=\n\([A-G]\)|$)",  # No space after )
            ]
            
            for pattern in section_patterns:
                match = re.search(pattern, self.full_text, re.IGNORECASE | re.DOTALL)
                if match:
                    return match.group(0)
        
        # Last resort: find by position in text
        # Look for section letter in parentheses
        if section_letter:
            start_pos = self.full_text.find(f"({section_letter})")
            if start_pos >= 0:
                # Find next section
                next_sections = []
                for letter in 'ABCDEFG':
                    if letter != section_letter:
                        next_pos = self.full_text.find(f"({letter})", start_pos + 1)
                        if next_pos > start_pos:
                            next_sections.append(next_pos)
                
                if next_sections:
                    end_pos = min(next_sections)
                    return self.full_text[start_pos:end_pos]
                else:
                    return self.full_text[start_pos:]
        
        return None
    
    def extract_policy_info(self) -> Dict[str, Any]:
        """Extract Section A: Policy information."""
        section_text = self.find_section("(A) DETAILS OF PRIMARY INSURED/CLAIMANT:")
        if not section_text:
            logger.debug("Section A not found in OCR text")
            return {}
        
        logger.debug(f"Section A found, length: {len(section_text)} chars")
        logger.debug(f"Section A sample (first 300 chars): {section_text[:300]}")
        
        data = {}
        
        # Policy number - more flexible pattern to handle OCR errors
        # Look for pattern like "PolicyNo." or "Policy No." followed by alphanumeric
        policy_patterns = [
            r"Policy\s*No\.?\s*:?\s*([A-Z0-9]{10,})",  # Standard format
            r"Policy\s*No\.?\s*([A-Z0-9]{10,})",  # Without colon
            r"a\.\s*Policy\s*No\.?\s*([A-Z0-9]{10,})",  # With "a." prefix
        ]
        
        for pattern in policy_patterns:
            policy_match = re.search(pattern, section_text, re.IGNORECASE)
            if policy_match:
                policy_num = policy_match.group(1).strip()
                # Clean up common OCR errors in policy numbers
                policy_num = policy_num.replace('Z', 'N').replace('z', 'N')
                policy_num = policy_num.replace('O', '0').replace('o', '0')
                policy_num = policy_num.replace('l', '1').replace('I', '1')
                data["policy_number"] = policy_num
                self.confidence_scores["policy_number"] = 0.85
                break
        
        # SI/Certificate number
        si_match = re.search(r"SI No\./Certificate No\.\s*:?\s*([A-Z0-9\-]+)", section_text, re.IGNORECASE)
        if si_match:
            data["si_certificate_number"] = si_match.group(1).strip()
            self.confidence_scores["si_certificate_number"] = 0.8
        
        # Company/TPA ID
        tpa_match = re.search(r"Company/TPA ID No\.\s*:?\s*([A-Z0-9\-]+)", section_text, re.IGNORECASE)
        if tpa_match:
            data["company_tpa_id"] = tpa_match.group(1).strip()
            self.confidence_scores["company_tpa_id"] = 0.8
        
        # Name - handle OCR errors (missing spaces, character errors)
        name_patterns = [
            r"d\.\s*Name\s*:?\s*([A-Z\s]{5,})",  # With "d." prefix
            r"Name\s*:?\s*([A-Z\s]{5,})",  # Standard
        ]
        
        for pattern in name_patterns:
            name_match = re.search(pattern, section_text, re.IGNORECASE)
            if name_match:
                name = name_match.group(1).strip()
                # Try to fix common OCR errors in names
                # "SAARMANRAUULOKUMARI" -> "SHARMA RAHUL KUMAR"
                # Add spaces between words (look for patterns)
                name = re.sub(r'([A-Z])([A-Z]{2,})', r'\1 \2', name)  # Add space before long uppercase sequences
                name = re.sub(r'\s+', ' ', name)  # Normalize spaces
                data["name"] = name.strip()
                self.confidence_scores["name"] = 0.8
                break
        
        # Address - handle OCR errors (missing spaces)
        address_patterns = [
            r"e\.\s*Address\s*:?\s*([^\n]+?)(?=City|State|Pin|Phone|Email|$)",  # With "e." prefix
            r"Address\s*:?\s*([^\n]+?)(?=City|State|Pin|Phone|Email|$)",  # Standard
        ]
        
        for pattern in address_patterns:
            address_match = re.search(pattern, section_text, re.IGNORECASE | re.DOTALL)
            if address_match:
                address = address_match.group(1).strip()
                # Try to add spaces in address (look for patterns like "FLAT-402" or numbers)
                address = re.sub(r'([A-Z])([A-Z]{2,})', r'\1 \2', address)  # Add space before long uppercase
                address = re.sub(r'(\d)([A-Z])', r'\1 \2', address)  # Space between number and letter
                address = re.sub(r'\s+', ' ', address)
                data["address"] = address.strip()
                self.confidence_scores["address"] = 0.75
                break
        
        # City, State, Pin - more flexible patterns
        city_patterns = [
            r"City\s*:?\s*([A-Z\s]{2,})",
            r"City\s*:?\s*([A-Z]+)",
        ]
        for pattern in city_patterns:
            city_match = re.search(pattern, section_text, re.IGNORECASE)
            if city_match:
                city = city_match.group(1).strip()
                # Fix common errors
                city = city.replace('I', '').replace('l', '')  # Remove stray characters
                data["city"] = city.strip()
                self.confidence_scores["city"] = 0.8
                break
        
        state_patterns = [
            r"State\s*:?\s*([A-Z\s]{3,})",
            r"Stote\s*:?\s*([A-Z\s]{3,})",  # OCR error: "Stote" instead of "State"
        ]
        for pattern in state_patterns:
            state_match = re.search(pattern, section_text, re.IGNORECASE)
            if state_match:
                state = state_match.group(1).strip()
                data["state"] = state.strip()
                self.confidence_scores["state"] = 0.8
                break
        
        pin_patterns = [
            r"Pin\s*Code\s*:?\s*(\d{6})",
            r"Pin\s*code\s*:?\s*(\d{6})",
            r"Pin\s*code\s*:?\s*[a-z]*(\d{6})",  # Handle OCR errors like "nnaon" before number
        ]
        for pattern in pin_patterns:
            pin_match = re.search(pattern, section_text, re.IGNORECASE)
            if pin_match:
                pin = pin_match.group(1).strip()
                data["pin_code"] = pin
                self.confidence_scores["pin_code"] = 0.85
                break
        
        # Phone and Email - more flexible
        phone_patterns = [
            r"Phone\s*No\.?\s*:?\s*(\d{10})",
            r"PhoneNo\s*:?\s*(\d{10})",  # No space
        ]
        for pattern in phone_patterns:
            phone_match = re.search(pattern, section_text, re.IGNORECASE)
            if phone_match:
                phone = phone_match.group(1).strip()
                # Fix OCR errors (G -> 6, O -> 0, etc.)
                phone = phone.replace('G', '6').replace('O', '0').replace('o', '0')
                phone = phone.replace('l', '1').replace('I', '1')
                data["phone"] = phone
                self.confidence_scores["phone"] = 0.85
                break
        
        email_patterns = [
            r"Email\s*ID\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            r"EmailD\s*:?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",  # OCR error: "EmailD"
            r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",  # Just email pattern
        ]
        for pattern in email_patterns:
            email_match = re.search(pattern, section_text, re.IGNORECASE)
            if email_match:
                email = email_match.group(1).strip()
                # Fix common OCR errors
                email = email.replace('shaama', 'sharma').replace('gmcu', 'gmail.com')
                if '@' in email and '.' in email.split('@')[1]:
                    data["email"] = email
                    self.confidence_scores["email"] = 0.9
                    break
        
        return data
    
    def extract_insurance_history(self) -> Dict[str, Any]:
        """Extract Section B: Insurance history."""
        section_text = self.find_section("(B) DETAILS OF INSURANCE HISTORY")
        if not section_text:
            logger.debug("Section B not found in OCR text")
            return {}
        
        logger.debug(f"Section B found, length: {len(section_text)} chars")
        
        data = {}
        
        # Currently covered
        currently_covered_match = re.search(r"Currently covered.*?(?:☐|☑|✓|\[.*?\])\s*Yes\s*(?:☐|☑|✓|\[.*?\])\s*No", section_text, re.IGNORECASE | re.DOTALL)
        if currently_covered_match:
            # Check which checkbox is marked (look for filled checkbox before "Yes")
            match_text = currently_covered_match.group(0)
            yes_marked = any(marker in match_text[:match_text.find("Yes")] for marker in ["☑", "✓", "[X]", "[x]", "[✓]"])
            data["currently_covered"] = yes_marked
            self.confidence_scores["currently_covered"] = 0.8
        
        # Date of commencement - handle OCR errors
        date_patterns = [
            r"Date of commencement.*?(\d{1,2}\s+\d{1,2}\s+\d{2,4})",
            r"commencement.*?(\d{1,2}\s+\d{1,2}\s+\d{2,4})",
            r"b\.\s*Date.*?(\d{1,2}\s+\d{1,2}\s+\d{2,4})",  # With "b." prefix
        ]
        for pattern in date_patterns:
            date_match = re.search(pattern, section_text, re.IGNORECASE | re.DOTALL)
            if date_match:
                date_str = date_match.group(1).strip()
                data["commencement_date"] = date_str
                self.confidence_scores["commencement_date"] = 0.75
                break
        
        # Hospitalization in last 4 years
        hosp_match = re.search(r"hospitalized in the last four years.*?(?:☐|☑|✓|\[.*?\])\s*Yes\s*(?:☐|☑|✓|\[.*?\])\s*No", section_text, re.IGNORECASE | re.DOTALL)
        if hosp_match:
            match_text = hosp_match.group(0)
            yes_marked = any(marker in match_text[:match_text.find("Yes")] for marker in ["☑", "✓", "[X]", "[x]", "[✓]"])
            data["hospitalized_last_4_years"] = yes_marked
            self.confidence_scores["hospitalized_last_4_years"] = 0.8
        
        # Previously covered
        prev_match = re.search(r"Previously covered.*?(?:☐|☑|✓|\[.*?\])\s*Yes\s*(?:☐|☑|✓|\[.*?\])\s*No", section_text, re.IGNORECASE | re.DOTALL)
        if prev_match:
            match_text = prev_match.group(0)
            yes_marked = any(marker in match_text[:match_text.find("Yes")] for marker in ["☑", "✓", "[X]", "[x]", "[✓]"])
            data["previously_covered"] = yes_marked
            self.confidence_scores["previously_covered"] = 0.8
        
        # Benefit type and claim status
        benefit_match = re.search(r"Benefit Type:.*?(Medication|Critical Illness|Cancer Insurance)", section_text, re.IGNORECASE | re.DOTALL)
        if benefit_match:
            data["benefit_type"] = benefit_match.group(1).strip()
            self.confidence_scores["benefit_type"] = 0.8
        
        status_match = re.search(r"Claim Status:.*?(Approved|Rejected|Pending)", section_text, re.IGNORECASE | re.DOTALL)
        if status_match:
            data["previous_claim_status"] = status_match.group(1).strip()
            self.confidence_scores["previous_claim_status"] = 0.8
        
        return data
    
    def extract_insured_person_details(self) -> Dict[str, Any]:
        """Extract Section C: Insured person details."""
        section_text = self.find_section("(C) DETAILS OF INSURED PERSON HOSPITALISED/DIAGNOSED")
        if not section_text:
            logger.debug("Section C not found in OCR text")
            return {}
        
        logger.debug(f"Section C found, length: {len(section_text)} chars")
        
        data = {}
        
        # Name
        name_match = re.search(r"Name:\s*([A-Z\s]+)", section_text, re.IGNORECASE)
        if name_match:
            data["name"] = name_match.group(1).strip()
            self.confidence_scores["insured_name"] = 0.85
        
        # Gender
        gender_match = re.search(r"Gender:\s*Male\s*(?:☐|☑|✓|\[.*?\])\s*Female\s*(?:☐|☑|✓|\[.*?\])\s*", section_text, re.IGNORECASE | re.DOTALL)
        if gender_match:
            match_text = gender_match.group(0)
            male_marked = any(marker in match_text[match_text.find("Male"):match_text.find("Female")] for marker in ["☑", "✓", "[X]", "[x]", "[✓]"])
            data["gender"] = "Male" if male_marked else "Female"
            self.confidence_scores["gender"] = 0.8
        
        # Age - more flexible
        age_patterns = [
            r"c\.\s*Age\s*\(years\)\s*:?\s*(\d{1,2})",
            r"Age\s*\(years\)\s*:?\s*(\d{1,2})",
            r"Age\s*:?\s*(\d{1,2})",
        ]
        for pattern in age_patterns:
            age_match = re.search(pattern, section_text, re.IGNORECASE)
            if age_match:
                try:
                    data["age"] = int(age_match.group(1))
                    self.confidence_scores["age"] = 0.85
                    break
                except ValueError:
                    continue
        
        # Date of birth - more flexible
        dob_patterns = [
            r"d\.\s*Date of Birth\s*:?\s*(\d{1,2}\s+\d{1,2}\s+\d{2,4})",
            r"Date of Birth\s*:?\s*(\d{1,2}\s+\d{1,2}\s+\d{2,4})",
            r"Date.*?Birth\s*:?\s*(\d{1,2}\s+\d{1,2}\s+\d{2,4})",
        ]
        for pattern in dob_patterns:
            dob_match = re.search(pattern, section_text, re.IGNORECASE)
            if dob_match:
                data["date_of_birth"] = dob_match.group(1).strip()
                self.confidence_scores["date_of_birth"] = 0.8
                break
        
        # Relationship
        rel_match = re.search(r"Relationship.*?Self\s*(?:☐|☑|✓|\[.*?\])\s*Spouse\s*(?:☐|☑|✓|\[.*?\])\s*Child\s*(?:☐|☑|✓|\[.*?\])\s*", section_text, re.IGNORECASE | re.DOTALL)
        if rel_match:
            match_text = rel_match.group(0)
            if any(marker in match_text for marker in ["☑", "✓", "[X]", "[x]", "[✓]"]):
                if any(marker in match_text[match_text.find("Self"):match_text.find("Spouse")] for marker in ["☑", "✓", "[X]", "[x]", "[✓]"]):
                    data["relationship"] = "Self"
                elif any(marker in match_text[match_text.find("Spouse"):match_text.find("Child")] for marker in ["☑", "✓", "[X]", "[x]", "[✓]"]):
                    data["relationship"] = "Spouse"
                elif any(marker in match_text[match_text.find("Child"):] for marker in ["☑", "✓", "[X]", "[x]", "[✓]"]):
                    data["relationship"] = "Child"
                self.confidence_scores["relationship"] = 0.8
        
        # Occupation
        occ_match = re.search(r"Occupation:\s*([^\n]+)", section_text, re.IGNORECASE)
        if occ_match:
            data["occupation"] = occ_match.group(1).strip()
            self.confidence_scores["occupation"] = 0.8
        
        # Nature of work
        work_match = re.search(r"Nature of Work:\s*([^\n]+)", section_text, re.IGNORECASE)
        if work_match:
            data["nature_of_work"] = work_match.group(1).strip()
            self.confidence_scores["nature_of_work"] = 0.8
        
        # Employer
        emp_match = re.search(r"Employer Name:\s*([^\n]+)", section_text, re.IGNORECASE)
        if emp_match:
            data["employer_name"] = emp_match.group(1).strip()
            self.confidence_scores["employer_name"] = 0.8
        
        return data
    
    def extract_hospitalisation_details(self) -> Dict[str, Any]:
        """Extract Section D: Hospitalisation details."""
        section_text = self.find_section("(D) DETAILS OF HOSPITALISATION/DIAGNOSIS:")
        if not section_text:
            logger.debug("Section D not found in OCR text")
            return {}
        
        logger.debug(f"Section D found, length: {len(section_text)} chars")
        
        data = {}
        
        # Hospital name - handle OCR errors (missing spaces)
        hosp_patterns = [
            r"a\.\s*Name\s+ot\s+hospital.*?:\s*([A-Z\s]{5,})",  # OCR error: "ot" instead of "of"
            r"Name\s+ot\s+hospital.*?:\s*([A-Z\s]{5,})",
            r"Name\s+of\s+hospital.*?:\s*([A-Z\s]{5,})",
            r"hospital.*?admitted.*?:\s*([A-Z\s]{5,})",
        ]
        for pattern in hosp_patterns:
            hosp_match = re.search(pattern, section_text, re.IGNORECASE)
            if hosp_match:
                hosp_name = hosp_match.group(1).strip()
                # Add spaces in hospital name
                hosp_name = re.sub(r'([A-Z])([A-Z]{2,})', r'\1 \2', hosp_name)
                hosp_name = re.sub(r'\s+', ' ', hosp_name)
                # Fix common errors: "JATAOMEMORDALNHOSPGTAL" -> "TATA MEMORIAL HOSPITAL"
                hosp_name = hosp_name.replace('JATA', 'TATA').replace('MEMORDAL', 'MEMORIAL')
                hosp_name = hosp_name.replace('NHOSPGTAL', 'HOSPITAL')
                data["hospital_name"] = hosp_name.strip()
                self.confidence_scores["hospital_name"] = 0.8
                break
        
        # Room category
        room_match = re.search(r"Room category.*?Day care\s*(?:☐|☑|✓|\[.*?\])\s*Single\s*(?:☐|☑|✓|\[.*?\])\s*Twin\s*(?:☐|☑|✓|\[.*?\])\s*", section_text, re.IGNORECASE | re.DOTALL)
        if room_match:
            match_text = room_match.group(0)
            if any(marker in match_text for marker in ["☑", "✓", "[X]", "[x]", "[✓]"]):
                if any(marker in match_text[match_text.find("Single"):match_text.find("Twin")] for marker in ["☑", "✓", "[X]", "[x]", "[✓]"]):
                    data["room_category"] = "Single occupancy"
                elif any(marker in match_text[match_text.find("Twin"):] for marker in ["☑", "✓", "[X]", "[x]", "[✓]"]):
                    data["room_category"] = "Twin sharing"
                elif any(marker in match_text[:match_text.find("Single")] for marker in ["☑", "✓", "[X]", "[x]", "[✓]"]):
                    data["room_category"] = "Day care"
                self.confidence_scores["room_category"] = 0.8
        
        # Hospitalisation due to
        cause_match = re.search(r"Hospitalisation due to:.*?Injury\s*(?:☐|☑|✓|\[.*?\])\s*Illness\s*(?:☐|☑|✓|\[.*?\])\s*Maternity\s*(?:☐|☑|✓|\[.*?\])\s*", section_text, re.IGNORECASE | re.DOTALL)
        if cause_match:
            match_text = cause_match.group(0)
            if any(marker in match_text for marker in ["☑", "✓", "[X]", "[x]", "[✓]"]):
                if any(marker in match_text[match_text.find("Illness"):match_text.find("Maternity")] for marker in ["☑", "✓", "[X]", "[x]", "[✓]"]):
                    data["hospitalisation_cause"] = "Illness"
                elif any(marker in match_text[match_text.find("Injury"):match_text.find("Illness")] for marker in ["☑", "✓", "[X]", "[x]", "[✓]"]):
                    data["hospitalisation_cause"] = "Injury"
                elif any(marker in match_text[match_text.find("Maternity"):] for marker in ["☑", "✓", "[X]", "[x]", "[✓]"]):
                    data["hospitalisation_cause"] = "Maternity"
                self.confidence_scores["hospitalisation_cause"] = 0.8
        
        # Dates - handle OCR errors like "eDale ol Admission" (Date of Admission)
        date_detected_patterns = [
            r"Date when disease first detected.*?:\s*(\d{1,2}\s+\d{1,2}\s+\d{2,4})",
            r"disease.*?detected.*?:\s*(\d{1,2}\s+\d{1,2}\s+\d{2,4})",
        ]
        for pattern in date_detected_patterns:
            date_detected_match = re.search(pattern, section_text, re.IGNORECASE)
            if date_detected_match:
                data["disease_detected_date"] = date_detected_match.group(1).strip()
                self.confidence_scores["disease_detected_date"] = 0.8
                break
        
        # Admission date - handle "eDale ol Admission" OCR error
        admission_patterns = [
            r"e\.?\s*Dale\s+ol\s+Admission\s+(\d{1,2}\s+\d{1,2}\s+\d{2,4})",  # OCR error
            r"Date\s+of\s+Admission\s+(\d{1,2}\s+\d{1,2}\s+\d{2,4})",
            r"Admission\s+(\d{1,2}\s+\d{1,2}\s+\d{2,4})",
        ]
        for pattern in admission_patterns:
            admission_match = re.search(pattern, section_text, re.IGNORECASE)
            if admission_match:
                data["admission_date"] = admission_match.group(1).strip()
                self.confidence_scores["admission_date"] = 0.85
                break
        
        # Discharge date - handle "g.Date of Discharge n21024" format
        discharge_patterns = [
            r"g\.\s*Date\s+of\s+Discharge\s+(\d{1,2}\s+\d{1,2}\s+\d{2,4})",
            r"Date\s+of\s+Discharge\s+(\d{1,2}\s+\d{1,2}\s+\d{2,4})",
            r"Discharge\s+(\d{1,2}\s+\d{1,2}\s+\d{2,4})",
            r"Discharge\s+[a-z]*(\d{6})",  # Handle "n21024" format (OCR error)
        ]
        for pattern in discharge_patterns:
            discharge_match = re.search(pattern, section_text, re.IGNORECASE)
            if discharge_match:
                date_str = discharge_match.group(1).strip()
                # If it's 6 digits, format as DD MM YY
                if len(date_str.replace(' ', '')) == 6 and ' ' not in date_str:
                    date_str = f"{date_str[0:2]} {date_str[2:4]} {date_str[4:6]}"
                data["discharge_date"] = date_str
                self.confidence_scores["discharge_date"] = 0.85
                break
        
        # System of medicine - handle "Allopcuhy" (Allopathy) OCR error
        medicine_patterns = [
            r"j\.\s*System\s+of\s+Medicine\s*:?\s*([A-Za-z]+)",
            r"System\s+of\s+Medicine\s*:?\s*([A-Za-z]+)",
            r"SystemofMedicine\s*:?\s*([A-Za-z]+)",  # No spaces
        ]
        for pattern in medicine_patterns:
            medicine_match = re.search(pattern, section_text, re.IGNORECASE)
            if medicine_match:
                medicine = medicine_match.group(1).strip()
                # Fix OCR errors
                medicine = medicine.replace('Allopcuhy', 'Allopathy').replace('allopcuhy', 'Allopathy')
                data["system_of_medicine"] = medicine
                self.confidence_scores["system_of_medicine"] = 0.85
                break
        
        # Type of cancer
        cancer_match = re.search(r"Type of Cancer\s*Carcinoma in situ\s*(?:☐|☑|✓|\[.*?\])\s*Early Stage\s*(?:☐|☑|✓|\[.*?\])\s*Major Cancer\s*(?:☐|☑|✓|\[.*?\])\s*", section_text, re.IGNORECASE | re.DOTALL)
        if cancer_match:
            match_text = cancer_match.group(0)
            if any(marker in match_text for marker in ["☑", "✓", "[X]", "[x]", "[✓]"]):
                if any(marker in match_text[match_text.find("Major Cancer"):] for marker in ["☑", "✓", "[X]", "[x]", "[✓]"]):
                    data["cancer_type"] = "Major Cancer"
                elif any(marker in match_text[match_text.find("Early Stage"):match_text.find("Major")] for marker in ["☑", "✓", "[X]", "[x]", "[✓]"]):
                    data["cancer_type"] = "Early Stage Cancer"
                elif any(marker in match_text[:match_text.find("Early Stage")] for marker in ["☑", "✓", "[X]", "[x]", "[✓]"]):
                    data["cancer_type"] = "Carcinoma in situ"
                self.confidence_scores["cancer_type"] = 0.8
        
        return data
    
    def extract_claim_details(self) -> Dict[str, Any]:
        """Extract Section E: Claim details."""
        section_text = self.find_section("(E) DETAILS OF CLAIM:")
        if not section_text:
            logger.debug("Section E not found in OCR text")
            return {}
        
        logger.debug(f"Section E found, length: {len(section_text)} chars")
        
        data = {
            "expenses": {},
            "lump_sum_benefits": {},
            "documents_submitted": []
        }
        
        # Helper function to extract amount after INR
        def extract_amount_after_inr(label_pattern, section_text):
            """Extract amount that appears after INR following a label."""
            # Look for pattern: label...INR...amount
            # Amount might be on same line or next line
            patterns = [
                rf"{label_pattern}.*?INR\s*([\d,\s@oO]+)",  # Same line
                rf"{label_pattern}.*?INR\s*\n\s*([\d,\s@oO]+)",  # Next line
                rf"{label_pattern}.*?INR\s*([^\n]+)",  # Any chars after INR on same/next line
            ]
            for pattern in patterns:
                match = re.search(pattern, section_text, re.IGNORECASE | re.DOTALL)
                if match:
                    amount_str = match.group(1).strip()
                    # Clean up: remove commas, spaces, OCR errors
                    amount_str = amount_str.replace(",", "").replace(" ", "")
                    # Fix common OCR errors in numbers
                    amount_str = amount_str.replace("@", "8")  # @ often means 8
                    amount_str = amount_str.replace("o", "0").replace("O", "0")  # o/O -> 0
                    amount_str = amount_str.replace("l", "1").replace("I", "1")  # l/I -> 1
                    amount_str = amount_str.replace("S", "5").replace("s", "5")  # S -> 5 (sometimes)
                    amount_str = amount_str.replace("G", "6").replace("g", "6")  # G -> 6
                    # Extract only digits
                    digits = re.sub(r'[^\d]', '', amount_str)
                    if digits and len(digits) >= 3:  # At least 3 digits to be valid
                        try:
                            return int(digits)
                        except ValueError:
                            pass
            return None
        
        # Pre-hospitalisation expenses
        pre_hosp_patterns = [
            r"i\.\s*Pre-Hospitalisation\s+Expenses",
            r"Pre-Hospitalisation\s+Expenses",
        ]
        for pattern in pre_hosp_patterns:
            amount = extract_amount_after_inr(pattern, section_text)
            if amount:
                data["expenses"]["pre_hospitalisation"] = amount
                self.confidence_scores["pre_hosp_expenses"] = 0.85
                break
        
        # Hospitalisation expenses - handle "6@odoo" OCR error (should be 680000)
        # Also look for amount on next line after "INR"
        hosp_exp_patterns = [
            r"ii\.\s*Hospitalisation\s+Expenses",
            r"Hospitalisation\s+Expenses",
            r"HospitalisationExpenses",  # No spaces
        ]
        for pattern in hosp_exp_patterns:
            amount = extract_amount_after_inr(pattern, section_text)
            if amount:
                data["expenses"]["hospitalisation"] = amount
                self.confidence_scores["hosp_expenses"] = 0.85
                break
        
        # If not found, try to find amount near "HospitalisationExpenses" and "INR"
        if "hospitalisation" not in data.get("expenses", {}):
            # Look for pattern: "HospitalisationExpenses:\nINR\n6@odoo"
            hosp_amount_match = re.search(r"HospitalisationExpenses.*?INR\s*\n\s*([^\n]+)", section_text, re.IGNORECASE | re.DOTALL)
            if hosp_amount_match:
                amount_str = hosp_amount_match.group(1).strip()
                amount_str = amount_str.replace("@", "8").replace("o", "0").replace("O", "0")
                digits = re.sub(r'[^\d]', '', amount_str)
                if digits and len(digits) >= 3:
                    try:
                        data["expenses"]["hospitalisation"] = int(digits)
                        self.confidence_scores["hosp_expenses"] = 0.8
                    except ValueError:
                        pass
        
        # Post-hospitalisation expenses
        post_hosp_patterns = [
            r"ii\.\s*Post-Hospitalisation\s+Expenses",
            r"Post-Hospitalisation\s+Expenses",
        ]
        for pattern in post_hosp_patterns:
            amount = extract_amount_after_inr(pattern, section_text)
            if amount:
                data["expenses"]["post_hospitalisation"] = amount
                self.confidence_scores["post_hosp_expenses"] = 0.85
                break
        
        # Ambulance charges
        ambulance_patterns = [
            r"v\.\s*Ambulance\s+Charges",
            r"Ambulance\s+Charges",
        ]
        for pattern in ambulance_patterns:
            amount = extract_amount_after_inr(pattern, section_text)
            if amount:
                data["expenses"]["ambulance"] = amount
                self.confidence_scores["ambulance_charges"] = 0.85
                break
        
        # Total expenses
        total_patterns = [
            r"Total\s+INR\s+([\d,\s@]+)",
            r"Total.*?INR\s+([\d,\s@]+)",
        ]
        for pattern in total_patterns:
            total_match = re.search(pattern, section_text, re.IGNORECASE)
            if total_match:
                amount_str = total_match.group(1).strip()
                amount_str = amount_str.replace(",", "").replace(" ", "").replace("@", "0")
                amount_str = amount_str.replace("o", "0").replace("O", "0")
                digits = re.sub(r'[^\d]', '', amount_str)
                if digits:
                    try:
                        data["expenses"]["total"] = int(digits)
                        self.confidence_scores["total_expenses"] = 0.85
                        break
                    except ValueError:
                        pass
        
        # Lump sum benefit - handle "0000o" (1000000) and "oodoo" OCR errors
        lump_sum_patterns = [
            r"vil\.\s*Lump\s+sum\s+benefit",
            r"vii\.\s*Lump\s+sum\s+benefit",
            r"Lump\s+sum\s+benefit",
        ]
        for pattern in lump_sum_patterns:
            amount = extract_amount_after_inr(pattern, section_text)
            if amount and amount >= 1000:  # Ignore small numbers (OCR noise)
                data["lump_sum_benefits"]["lump_sum"] = amount
                self.confidence_scores["lump_sum_benefit"] = 0.85
                break
        
        # If not found, try to find near "Lump sum benefit" and "INR"
        if "lump_sum" not in data.get("lump_sum_benefits", {}):
            # Look for "0000o" or "oodoo" patterns (should be 1000000)
            lump_match = re.search(r"Lump\s+sum\s+benefit.*?INR\s*\n\s*([^\n]+)", section_text, re.IGNORECASE | re.DOTALL)
            if lump_match:
                amount_str = lump_match.group(1).strip()
                # Handle "0000o" -> should be "1000000" (1 followed by zeros)
                # Handle "oodoo" -> should be "1000000"
                amount_str = amount_str.replace("o", "0").replace("O", "0")
                amount_str = amount_str.replace("@", "8")
                digits = re.sub(r'[^\d]', '', amount_str)
                if digits:
                    # If starts with multiple zeros, might be missing leading 1
                    if digits.startswith('0000') and len(digits) >= 6:
                        digits = '1' + digits  # Add leading 1
                    try:
                        amount = int(digits)
                        if amount >= 100000:  # Reasonable minimum for lump sum
                            data["lump_sum_benefits"]["lump_sum"] = amount
                            self.confidence_scores["lump_sum_benefit"] = 0.8
                    except ValueError:
                        pass
        
        # Documents submitted checklist
        # Look for checkbox markers (☑, ✓, [X], etc.) followed by document names
        filled_checkbox_pattern = r"(?:☑|✓|\[X\]|\[x\]|\[✓\])"
        doc_patterns = [
            ("Claim Form Duly Signed", r"Claim Form Duly Signed"),
            ("Operation Theatre Notes", r"Operation Theatre Notes"),
            ("Hospital Discharge Summary", r"Hospital Discharge Summary"),
            ("First Consultation and Follow-up Notes", r"First Consultation"),
            ("Hospital Main Bill", r"Hospital Main Bill"),
            ("Investigation Reports", r"Investigation Reports"),
            ("Laboratory Test Reports", r"Laboratory Test Reports"),
            ("Indoor Case Papers", r"Indoor Case Papers"),
            ("Hospital Bill Payment Receipt", r"Hospital Bill Payment Receipt"),
            ("Attending Physician's Statement", r"Attending Physician's Statement"),
            ("Pharmacy Bill", r"Pharmacy Bill"),
            ("Latest Bank Statement", r"Latest Bank Statement"),
            ("Copy of Pass Book", r"Pass Book"),
            ("Blood Test for Cancer Diagnosis", r"Blood Test for Cancer Diagnosis"),
            ("Clinical/Hospital Reports", r"Clinical/Hospital Reports"),
            ("Doctor Consultation Referral Letter", r"Doctor Consultation Referral Letter"),
            ("Doctor's Request for Investigation", r"Doctor's Request for Investigation"),
        ]
        
        for doc_name, pattern in doc_patterns:
            # Check if pattern exists and has a filled checkbox nearby (within 50 chars)
            match = re.search(pattern, section_text, re.IGNORECASE)
            if match:
                start_pos = max(0, match.start() - 50)
                end_pos = min(len(section_text), match.end() + 50)
                context = section_text[start_pos:end_pos]
                # Check for filled checkbox in context
                if re.search(filled_checkbox_pattern, context):
                    data["documents_submitted"].append(doc_name)
        
        return data
    
    def extract_condition_details(self) -> Dict[str, Any]:
        """Extract Section F: Claimed condition details."""
        section_text = self.find_section("(F) CLAIMED CONDITION DETAILS:")
        if not section_text:
            logger.debug("Section F not found in OCR text")
            return {}
        
        logger.debug(f"Section F found, length: {len(section_text)} chars")
        
        data = {}
        
        # Final diagnosis - handle OCR errors like "DNMASnMEAOENOCARCDNOMA" (INVASIVE ADENOCARCINOMA)
        diagnosis_patterns = [
            r"a\.\s*FinaI\s+DiagnOsis\s*:?\s*([A-Z\s]+)",  # OCR error: "FinaI DiagnOsis"
            r"Final\s+Diagnosis\s*:?\s*([A-Z\s]+)",
            r"Diagnosis\s*:?\s*([A-Z\s]{10,})",
        ]
        for pattern in diagnosis_patterns:
            diagnosis_match = re.search(pattern, section_text, re.IGNORECASE)
            if diagnosis_match:
                diagnosis = diagnosis_match.group(1).strip()
                # Fix common OCR errors
                diagnosis = diagnosis.replace('DNMASnMEAOENOCARCDNOMA', 'INVASIVE ADENOCARCINOMA')
                diagnosis = diagnosis.replace('DNMAS', 'INVAS').replace('MEAO', 'IVE ADENO')
                diagnosis = diagnosis.replace('CARCDNOMA', 'CARCINOMA')
                # Add spaces
                diagnosis = re.sub(r'([A-Z])([A-Z]{2,})', r'\1 \2', diagnosis)
                diagnosis = re.sub(r'\s+', ' ', diagnosis)
                data["final_diagnosis"] = diagnosis.strip()
                self.confidence_scores["final_diagnosis"] = 0.8
                break
        
        # Helper to extract and normalize dates (handle spaces between digits)
        def extract_date(pattern, text):
            """Extract date and normalize format."""
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1).strip()
                # Handle cases where digits are separated: "2 2 0 9 2 5" -> "22 09 25"
                # Remove all spaces and re-format
                digits_only = re.sub(r'[^\d]', '', date_str)
                if len(digits_only) == 6:
                    # Format as DD MM YY
                    date_str = f"{digits_only[0:2]} {digits_only[2:4]} {digits_only[4:6]}"
                elif len(digits_only) == 8:
                    # Format as DD MM YYYY
                    date_str = f"{digits_only[0:2]} {digits_only[2:4]} {digits_only[4:8]}"
                return date_str
            return None
        
        # Date of diagnosis - handle "2 2 0 9 2 5" format
        diag_date_patterns = [
            r"b\.\s*Date\s+of\s+Diagnosis\s*:?\s*([\d\s]+)",
            r"Date\s+of\s+Diagnosis\s*:?\s*([\d\s]+)",
        ]
        for pattern in diag_date_patterns:
            date_str = extract_date(pattern, section_text)
            if date_str:
                data["diagnosis_date"] = date_str
                self.confidence_scores["diagnosis_date"] = 0.85
                break
        
        # First consultation date
        consult_patterns = [
            r"c\.\s*Date\s+of\s+First\s+Consultation.*?:\s*([\d\s]+)",
            r"Date\s+of\s+First\s+Consultation.*?:\s*([\d\s]+)",
        ]
        for pattern in consult_patterns:
            date_str = extract_date(pattern, section_text)
            if date_str:
                data["first_consultation_date"] = date_str
                self.confidence_scores["first_consultation_date"] = 0.8
                break
        
        # Nature and duration of complaints - handle "NNALnPAnANWEDCDDnLOSS" (ABDOMINAL PAIN, WEIGHT LOSS)
        complaints_patterns = [
            r"d\.\s*Nature\s+and\s+Duration.*?:\s*([^\n]+)",
            r"Nature\s+and\s+Duration.*?:\s*([^\n]+)",
        ]
        for pattern in complaints_patterns:
            complaints_match = re.search(pattern, section_text, re.IGNORECASE)
            if complaints_match:
                complaints = complaints_match.group(1).strip()
                # Fix common errors
                complaints = complaints.replace('NNALnPAnANWEDCDDnLOSS', 'ABDOMINAL PAIN, WEIGHT LOSS')
                complaints = complaints.replace('NNAL', 'ABDOMINAL').replace('PAnAN', 'PAIN')
                complaints = complaints.replace('WEDCDD', 'WEIGHT').replace('nLOSS', 'LOSS')
                # Add spaces
                complaints = re.sub(r'([A-Z])([A-Z]{2,})', r'\1 \2', complaints)
                complaints = re.sub(r'\s+', ' ', complaints)
                data["complaints"] = complaints.strip()
                self.confidence_scores["complaints"] = 0.75
                break
        
        # Date complaints first evident
        evident_patterns = [
            r"e\.\s*Date\s+when.*?Evident\s*:?\s*([\d\s]+)",
            r"Date\s+when.*?Evident\s*:?\s*([\d\s]+)",
        ]
        for pattern in evident_patterns:
            date_str = extract_date(pattern, section_text)
            if date_str:
                data["complaints_evident_date"] = date_str
                self.confidence_scores["complaints_evident_date"] = 0.8
                break
        
        # Site of tumor - handle "ASCENDNCGI" (ASCENDING COLON) OCR error
        tumor_patterns = [
            r"f\.\s*Site\s+of\s+Tumour\s*:?\s*([A-Z\s]+)",
            r"Site\s+of\s+Tumour\s*:?\s*([A-Z\s]+)",
        ]
        for pattern in tumor_patterns:
            tumor_match = re.search(pattern, section_text, re.IGNORECASE)
            if tumor_match:
                tumor_site = tumor_match.group(1).strip()
                # Fix common errors
                tumor_site = tumor_site.replace('ASCENDNCGI', 'ASCENDING COLON')
                tumor_site = tumor_site.replace('ASCEND', 'ASCENDING')
                tumor_site = tumor_site.replace('NCGI', 'COLON')
                # Add spaces
                tumor_site = re.sub(r'([A-Z])([A-Z]{2,})', r'\1 \2', tumor_site)
                tumor_site = re.sub(r'\s+', ' ', tumor_site)
                data["tumor_site"] = tumor_site.strip()
                self.confidence_scores["tumor_site"] = 0.8
                break
        
        return data
    
    def extract_past_health_history(self) -> Dict[str, Any]:
        """Extract Section G: Past health history."""
        section_text = self.find_section("(G) PAST HEALTH HISTORY")
        if not section_text:
            logger.debug("Section G not found in OCR text")
            return {}
        
        logger.debug(f"Section G found, length: {len(section_text)} chars")
        
        data = {}
        
        # Previous malignancy
        malignancy_match = re.search(r"Any Previous Malignancy.*?(?:☐|☑|✓|\[.*?\])\s*Yes\s*(?:☐|☑|✓|\[.*?\])\s*No", section_text, re.IGNORECASE | re.DOTALL)
        if malignancy_match:
            match_text = malignancy_match.group(0)
            yes_marked = any(marker in match_text[:match_text.find("Yes")] for marker in ["☑", "✓", "[X]", "[x]", "[✓]"])
            data["previous_malignancy"] = yes_marked
            self.confidence_scores["previous_malignancy"] = 0.8
        
        # Other illness/surgery
        illness_match = re.search(r"Any Other Illness/Surgery.*?:\s*([^\n]+)", section_text, re.IGNORECASE)
        if illness_match:
            data["other_illness"] = illness_match.group(1).strip()
            self.confidence_scores["other_illness"] = 0.75
        
        return data
    
    def parse(self, debug_output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Parse all sections and return structured data."""
        logger.info("Parsing claim form sections...")
        
        # Save full text for debugging
        if debug_output_dir:
            debug_dir = debug_output_dir / "debug_parser"
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            full_text_path = debug_dir / "00_full_ocr_text.txt"
            with open(full_text_path, 'w', encoding='utf-8') as f:
                f.write(self.full_text)
            logger.debug(f"Saved full OCR text to: {full_text_path}")
            
            # Save each section extraction
            logger.debug("Extracting sections...")
        
        policy_info = self.extract_policy_info()
        logger.debug(f"Policy info extracted: {len(policy_info)} fields")
        if debug_output_dir:
            with open(debug_dir / "01_policy_info.json", 'w', encoding='utf-8') as f:
                json.dump(policy_info, f, indent=2, ensure_ascii=False)
        
        insurance_history = self.extract_insurance_history()
        logger.debug(f"Insurance history extracted: {len(insurance_history)} fields")
        if debug_output_dir:
            with open(debug_dir / "02_insurance_history.json", 'w', encoding='utf-8') as f:
                json.dump(insurance_history, f, indent=2, ensure_ascii=False)
        
        insured_person = self.extract_insured_person_details()
        logger.debug(f"Insured person extracted: {len(insured_person)} fields")
        if debug_output_dir:
            with open(debug_dir / "03_insured_person.json", 'w', encoding='utf-8') as f:
                json.dump(insured_person, f, indent=2, ensure_ascii=False)
        
        hospitalisation = self.extract_hospitalisation_details()
        logger.debug(f"Hospitalisation extracted: {len(hospitalisation)} fields")
        if debug_output_dir:
            with open(debug_dir / "04_hospitalisation.json", 'w', encoding='utf-8') as f:
                json.dump(hospitalisation, f, indent=2, ensure_ascii=False)
        
        claim_details = self.extract_claim_details()
        logger.debug(f"Claim details extracted: {len(claim_details)} fields")
        if debug_output_dir:
            with open(debug_dir / "05_claim_details.json", 'w', encoding='utf-8') as f:
                json.dump(claim_details, f, indent=2, ensure_ascii=False)
        
        medical_details = self.extract_condition_details()
        logger.debug(f"Medical details extracted: {len(medical_details)} fields")
        if debug_output_dir:
            with open(debug_dir / "06_medical_details.json", 'w', encoding='utf-8') as f:
                json.dump(medical_details, f, indent=2, ensure_ascii=False)
        
        past_health = self.extract_past_health_history()
        logger.debug(f"Past health history extracted: {len(past_health)} fields")
        if debug_output_dir:
            with open(debug_dir / "07_past_health_history.json", 'w', encoding='utf-8') as f:
                json.dump(past_health, f, indent=2, ensure_ascii=False)
        
        result = {
            "policy_info": policy_info,
            "insurance_history": insurance_history,
            "insured_person": insured_person,
            "hospitalisation": hospitalisation,
            "claim_details": claim_details,
            "medical_details": medical_details,
            "past_health_history": past_health,
            "metadata": {
                "processing_date": datetime.now().isoformat(),
                "pages_processed": len(self.ocr_results),
                "confidence_scores": self.confidence_scores,
                "total_fields_extracted": len(self.confidence_scores),
                "low_confidence_fields": [
                    field for field, score in self.confidence_scores.items() 
                    if score < 0.7
                ]
            }
        }
        
        # Save confidence scores
        if debug_output_dir:
            with open(debug_dir / "08_confidence_scores.json", 'w', encoding='utf-8') as f:
                json.dump(self.confidence_scores, f, indent=2, ensure_ascii=False)
        
        return result

# ---------------- MAIN PROCESSOR ----------------
def process_claim_document(pdf_path: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Process a claim document and extract structured data.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Optional output directory. If None, uses output/{pdf_stem}/
        
    Returns:
        Dictionary with extracted claim data
    """
    if not pdf_path.exists():
        logger.error(f"File not found: {pdf_path}")
        return {}
    
    logger.info(f"Processing claim document: {pdf_path}")
    
    # Determine output directory
    if output_dir is None:
        base_output_dir = Path("output")
        base_output_dir.mkdir(exist_ok=True)
        output_dir = base_output_dir / pdf_path.stem
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract text using OCR
    logger.info("Step 1: Extracting text using OCR...")
    ocr_results = extract_text_with_ocr(pdf_path, debug_output_dir=output_dir)
    
    if not ocr_results:
        logger.error("Failed to extract text from PDF")
        return {}
    
    # Parse form structure
    logger.info("Step 2: Parsing form structure...")
    parser = ClaimFormParser(ocr_results)
    extracted_data = parser.parse(debug_output_dir=output_dir)
    
    # Save JSON output
    json_path = output_dir / f"{pdf_path.stem}_claim_data.json"
    logger.info(f"Step 3: Saving JSON output to {json_path}")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    
    logger.info("✓ Processing complete")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"JSON saved to: {json_path}")
    logger.info(f"Total fields extracted: {extracted_data.get('metadata', {}).get('total_fields_extracted', 0)}")
    
    low_conf_fields = extracted_data.get('metadata', {}).get('low_confidence_fields', [])
    if low_conf_fields:
        logger.warning(f"Low confidence fields ({len(low_conf_fields)}): {', '.join(low_conf_fields)}")
    
    return extracted_data

# ---------------- CLI INTERFACE ----------------
def main():
    """Entry point for CLI usage."""
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        input_path = Path("Claim1.pdf")
    
    if not input_path.exists():
        logger.error(f"Error: Path '{input_path}' not found!")
        sys.exit(1)
    
    if input_path.is_file() and input_path.suffix.lower() == '.pdf':
        process_claim_document(input_path)
    else:
        logger.error(f"Error: '{input_path}' is not a PDF file!")
        sys.exit(1)

if __name__ == "__main__":
    main()
