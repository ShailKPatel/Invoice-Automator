"""
Standalone extractor: call extract_text(path: str) -> str
Returns the extracted text (string) or raises an exception on fatal failure.

01 - extract text

Notes:
  - This function prefers native PDF text extraction via PyMuPDF.
  - If native text is short (<10 chars), it falls back to EasyOCR rasterization.
  - For images it uses EasyOCR directly after lightweight preprocessing.
  - The function returns only the extracted text (str).
"""
from pathlib import Path
from typing import Optional, List
import tempfile
import statistics
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import easyocr

# Configuration
PDF_TEXT_MIN_CHARS = 10
MIN_OCR_CONFIDENCE = 0.50
# Initialize EasyOCR (English). 
ocr_reader = easyocr.Reader(["en"], gpu=False)


def _is_text_meaningful(text: Optional[str], min_chars: int = PDF_TEXT_MIN_CHARS) -> bool:
    if not text:
        return False
    return len(text.strip()) >= min_chars


def _parse_easyocr_result(raw_result) -> (str, Optional[float]):
    """Combine EasyOCR outputs into a single string and return mean confidence."""
    if not raw_result:
        return "", None
    lines: List[str] = []
    confs: List[float] = []
    for item in raw_result:
        # Typical item: [bbox, text, confidence]
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                text = str(item[1]).strip()
                conf = float(item[2])
                if text:
                    lines.append(text)
                confs.append(conf)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("string")
                conf = item.get("confidence") or item.get("score")
                if text:
                    lines.append(str(text).strip())
                if conf is not None:
                    confs.append(float(conf))
            elif isinstance(item, str):
                t = item.strip()
                if t:
                    lines.append(t)
        except Exception:
            continue
    final_text = "\n".join([" ".join(l.split()) for l in lines]).strip()
    mean_conf = float(statistics.mean(confs)) if confs else None
    return final_text, mean_conf


def _preprocess_image_for_ocr(pil_image: Image.Image) -> Image.Image:
    """Return a preprocessed PIL Image for OCR (grayscale, enhance, denoise via threshold)."""
    img = pil_image.convert("L")
    img = ImageEnhance.Brightness(img).enhance(1.15)
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Sharpness(img).enhance(1.5)
    arr = np.array(img)
    # Otsu threshold
    _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(th)


def _ocr_image_path(image_path: Path) -> (str, Optional[float]):
    """Run EasyOCR on an image file path. Returns (text, mean_conf)."""
    raw = ocr_reader.readtext(str(image_path))
    text, conf = _parse_easyocr_result(raw)
    return text, conf


def _ocr_from_pil_images(images: List[Image.Image]) -> (str, Optional[float]):
    """OCR a list of PIL images and return the concatenated text and mean confidence."""
    texts: List[str] = []
    confs: List[float] = []
    with tempfile.TemporaryDirectory() as td:
        for i, img in enumerate(images, start=1):
            pre = _preprocess_image_for_ocr(img)
            p = Path(td) / f"page_{i}.png"
            pre.save(p, "PNG")
            t, c = _ocr_image_path(p)
            if t:
                texts.append(t)
            if c is not None:
                confs.append(c)
    agg_text = "\n".join(texts).strip()
    mean_conf = float(statistics.mean(confs)) if confs else None
    return agg_text, mean_conf


def extract_text(path: str) -> str:
    """Main entry. Accepts a file path (PDF or image). Returns extracted text string.

    Raises FileNotFoundError, RuntimeError, or ValueError on failure.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    suffix = p.suffix.lower()

    # PDF path: try native extraction first
    if suffix == ".pdf":
        native_parts: List[str] = []
        try:
            with fitz.open(str(p)) as doc:
                for page in doc:
                    native_parts.append(page.get_text("text") or "")
        except Exception as e:
            native_parts = []
        native_text = "\n".join(native_parts).strip()
        if _is_text_meaningful(native_text):
            return native_text
        # Fallback: rasterize pages and OCR
        try:
            pil_pages = convert_from_path(str(p), dpi=300)
        except Exception as e:
            raise RuntimeError(f"Failed to rasterize PDF for OCR: {e}")
        ocr_text, mean_conf = _ocr_from_pil_images(pil_pages)
        if not _is_text_meaningful(ocr_text):
            raise ValueError("No meaningful text extracted after OCR fallback")
        if mean_conf is not None and mean_conf < MIN_OCR_CONFIDENCE:
            raise ValueError(f"Low OCR confidence: {mean_conf:.4f}")
        return ocr_text

    # Image path: open and OCR
    else:
        try:
            with Image.open(p) as im:
                im_rgb = im.convert("RGB")
                ocr_text, mean_conf = _ocr_from_pil_images([im_rgb])
        except Exception as e:
            raise RuntimeError(f"Failed to open or OCR image: {e}")
        if not _is_text_meaningful(ocr_text):
            raise ValueError("No meaningful text found in image")
        if mean_conf is not None and mean_conf < MIN_OCR_CONFIDENCE:
            raise ValueError(f"Low OCR confidence: {mean_conf:.4f}")
        return ocr_text


# Example usage 
if __name__ == '__main__':
    print(extract_text(r'.\data\raw\train\Copy of ARPFIINVOEBTCHLASER (1).pdf'))
