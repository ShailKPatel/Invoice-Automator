"""
Standalone extractor: call extract_text(path_or_file) -> str
Returns the extracted text (string) or raises an exception on fatal failure.

Features and compatibility notes:
 - Works with PDF files and image files.
 - Accepts:
     * a filesystem path (str or pathlib.Path)
     * bytes
     * a file-like object with read()
     * a Streamlit UploadedFile object (best-effort detection)
 - Prefers native text extraction via PyMuPDF (fitz). If native text is too short,
   falls back to rasterizing pages with pdf2image and running EasyOCR on images.
 - For images, runs EasyOCR after lightweight preprocessing.
 - Path handling is normalized so Windows backslashes do not cause failures on Linux.
 - Heavy initializations (EasyOCR reader) are lazy and cached.
 - Adds robust logging and clear exception messages suitable for both local and hosted use.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import logging
import statistics
from pathlib import Path
from typing import Optional, List, Tuple, Union, Any

# Third-party imports
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # will check at runtime

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None  # will check at runtime

try:
    from PIL import Image, ImageEnhance
except Exception:
    Image = None  # will check at runtime

try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import easyocr
except Exception:
    easyocr = None

# Configuration
PDF_TEXT_MIN_CHARS = 10
MIN_OCR_CONFIDENCE = 0.50
OCR_LANGUAGES = ["en"]  # adjust if you need extra languages
OCR_GPU = False  # set to True if the environment has a GPU and easyocr built with cuda
OCR_INIT_TIMEOUT_SEC = 20.0  # not enforced, informational

# Logging
logger = logging.getLogger("extractor")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


# Lazy-initialized OCR reader instance
_OCR_READER: Optional[Any] = None


def _get_ocr_reader() -> Any:
    """
    Initialize and cache the EasyOCR reader. This is lazy because building the reader
    can be slow and memory heavy on import time.
    """
    global _OCR_READER
    if _OCR_READER is not None:
        return _OCR_READER
    if easyocr is None:
        raise RuntimeError("easyocr is required but not installed in the environment")
    logger.debug("Initializing EasyOCR reader (languages=%s, gpu=%s)", OCR_LANGUAGES, OCR_GPU)
    _OCR_READER = easyocr.Reader(OCR_LANGUAGES, gpu=OCR_GPU)
    return _OCR_READER


def _is_text_meaningful(text: Optional[str], min_chars: int = PDF_TEXT_MIN_CHARS) -> bool:
    """Return True if the text is non-empty and meets a minimum character threshold."""
    if not text:
        return False
    return len(text.strip()) >= int(min_chars)


def _parse_easyocr_result(raw_result: Any) -> Tuple[str, Optional[float]]:
    """
    Combine EasyOCR outputs into a single string and return mean confidence.

    Accepts EasyOCR raw result formats such as list of [bbox, text, confidence]
    or list of dicts with 'text'/'confidence' keys.
    """
    if not raw_result:
        return "", None
    lines: List[str] = []
    confs: List[float] = []
    for item in raw_result:
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                text = str(item[1]).strip()
                try:
                    conf = float(item[2])
                except Exception:
                    conf = None
                if text:
                    lines.append(text)
                if conf is not None:
                    confs.append(conf)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("string") or ""
                conf = item.get("confidence") or item.get("score")
                if text:
                    lines.append(str(text).strip())
                if conf is not None:
                    try:
                        confs.append(float(conf))
                    except Exception:
                        pass
            elif isinstance(item, str):
                t = item.strip()
                if t:
                    lines.append(t)
        except Exception:
            continue
    # Normalize whitespace inside lines, then join with newlines
    final_text = "\n".join([" ".join(l.split()) for l in lines]).strip()
    mean_conf = float(statistics.mean(confs)) if confs else None
    return final_text, mean_conf


def _preprocess_image_for_ocr(pil_image: "Image.Image") -> "Image.Image":
    """
    Preprocess PIL Image for better OCR results:
      - convert to grayscale
      - slight brightness/contrast/sharpness enhance
      - Otsu thresholding if cv2 available
    """
    if Image is None:
        raise RuntimeError("Pillow (PIL) is required for image preprocessing")
    img = pil_image.convert("L")
    try:
        img = ImageEnhance.Brightness(img).enhance(1.15)
        img = ImageEnhance.Contrast(img).enhance(1.25)
        img = ImageEnhance.Sharpness(img).enhance(1.5)
    except Exception:
        # Enhancement is optional; continue even if it fails for some reason
        pass
    if np is None or cv2 is None:
        # If opencv/numpy not available, return the grayscaled enhanced image
        return img
    arr = np.array(img)
    try:
        _, th = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(th)
    except Exception:
        return img


def _ocr_image_path(image_path: Path) -> Tuple[str, Optional[float]]:
    """
    Run EasyOCR on an image file path. Returns (text, mean_confidence).
    """
    reader = _get_ocr_reader()
    # easyocr accepts str path
    raw = reader.readtext(str(image_path))
    return _parse_easyocr_result(raw)


def _ocr_from_pil_images(images: List["Image.Image"]) -> Tuple[str, Optional[float]]:
    """
    OCR a list of PIL images and return the concatenated text and mean confidence.
    Saves each preprocessed image temporarily to disk because easyocr expects a path
    or numpy array. Using temporary files keeps memory lower for large PDFs.
    """
    if not images:
        return "", None
    texts: List[str] = []
    confs: List[float] = []
    with tempfile.TemporaryDirectory() as td:
        for i, img in enumerate(images, start=1):
            try:
                pre = _preprocess_image_for_ocr(img)
            except Exception as e:
                logger.debug("Image preprocessing failed for page %d: %s", i, e)
                pre = img
            p = Path(td) / f"page_{i}.png"
            try:
                pre.save(p, "PNG")
            except Exception as e:
                # fallback: try converting to RGB and save
                try:
                    pre.convert("RGB").save(p, "PNG")
                except Exception as e2:
                    logger.debug("Failed to save preprocessed page %d: %s; %s", i, e, e2)
                    continue
            try:
                t, c = _ocr_image_path(p)
            except Exception as e:
                logger.debug("EasyOCR failed on page %d (path=%s): %s", i, p, e)
                t, c = "", None
            if t:
                texts.append(t)
            if c is not None:
                confs.append(c)
    agg_text = "\n".join(texts).strip()
    mean_conf = float(statistics.mean(confs)) if confs else None
    return agg_text, mean_conf


def _save_bytes_to_tempfile(data: bytes, suffix: str = "") -> Path:
    """
    Save raw bytes to a temporary file and return the pathlib.Path to it.
    Caller is responsible for deleting the file if desired.
    """
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tf.write(data)
        tf.flush()
        return Path(tf.name)
    finally:
        tf.close()


def _normalize_and_resolve_path(path_str: str) -> Path:
    """
    Normalize a path string so Windows backslashes do not break Linux deployments.
    Attempts to resolve relative paths against cwd and module file location if needed.
    Does not create files, only tries to find a sensible existing path.
    """
    if not isinstance(path_str, str):
        raise TypeError("path_str must be a string")

    # Replace backslashes with forward slashes to avoid escape issues
    candidate = Path(path_str.replace("\\", "/"))

    # If user passed a drive letter like 'C:/' this will not exist on Linux
    if candidate.drive and not candidate.exists():
        # Try removing the drive letter; sometimes paths were copied from Windows and only the relative
        # tail exists in the current repo. For safety, log and continue.
        logger.debug("Path includes a drive (%s) and does not exist on host: %s", candidate.drive, path_str)
        # Try using the tail after the drive
        tail = Path(*candidate.parts[1:]) if len(candidate.parts) > 1 else candidate.name
        candidate2 = Path.cwd() / tail
        if candidate2.exists():
            return candidate2.resolve()
        # Fallthrough: return the original Path object (which may not exist)
        return candidate

    # If path is relative, try resolving against cwd and script directory
    if not candidate.is_absolute():
        # 1) cwd
        cand1 = (Path.cwd() / candidate).resolve()
        if cand1.exists():
            return cand1
        # 2) project relative to this file
        this_file_dir = Path(__file__).resolve().parent
        cand2 = (this_file_dir / candidate).resolve()
        if cand2.exists():
            return cand2
        # 3) return as-is (may not exist)
        return candidate.resolve()
    # absolute path
    return candidate.resolve()


def _rasterize_pdf_to_pil_images(pdf_path: Path, dpi: int = 300) -> List["Image.Image"]:
    """
    Rasterize a PDF file to a list of PIL images using pdf2image.convert_from_path.
    Raises RuntimeError with a helpful message if rasterization fails.
    Note: pdf2image requires poppler to be installed on the host system.
    """
    if convert_from_path is None:
        raise RuntimeError(
            "pdf2image is required for PDF rasterization but is not installed in the environment"
        )
    try:
        # convert_from_path returns list of PIL images
        images = convert_from_path(str(pdf_path), dpi=dpi)
        if not images:
            raise RuntimeError("pdf2image produced no pages for rasterization")
        return images
    except Exception as e:
        # Provide a clear hint about poppler on Linux/Streamlit hosts
        msg = str(e)
        if "Poppler" in msg or "pdftoppm" in msg or "pdfinfo" in msg:
            raise RuntimeError(
                f"Failed to rasterize PDF. pdf2image/poppler error: {e}. "
                "On Linux hosts you may need to install poppler-utils (e.g. apt-get install poppler-utils)."
            )
        raise RuntimeError(f"Failed to rasterize PDF: {e}")


def extract_text(path_or_file: Union[str, Path, bytes, io.IOBase, Any]) -> str:
    """
    Main entry. Accepts a filesystem path (string or Path), bytes, or file-like object.
    Returns extracted text string or raises FileNotFoundError, RuntimeError, ValueError.

    Behavior:
      - If a Path/str to a PDF is provided, attempt native extraction via PyMuPDF.
        If native text is short, fallback to rasterize + OCR.
      - If bytes or file-like are provided, save to a temporary file and process.
    """
    # Detect streamlit UploadedFile by duck-typing if present
    uploaded_file = None
    try:
        # Streamlit's UploadedFile has getbuffer method and is file-like
        if hasattr(path_or_file, "getbuffer") and callable(getattr(path_or_file, "getbuffer")):
            uploaded_file = path_or_file
    except Exception:
        uploaded_file = None

    # If bytes, save to temp file
    temp_file_to_cleanup: Optional[Path] = None
    try:
        if isinstance(path_or_file, (bytes, bytearray)):
            # Need to write to temp file, try to detect likely suffix by examining header
            suffix = ""
            header = bytes(path_or_file[:8])
            if header.startswith(b"%PDF"):
                suffix = ".pdf"
            else:
                suffix = ".png"
            temp_path = _save_bytes_to_tempfile(bytes(path_or_file), suffix=suffix)
            temp_file_to_cleanup = temp_path
            p = temp_path
        elif uploaded_file is not None:
            # Streamlit UploadedFile: write buffer to temp file
            data = uploaded_file.getbuffer()
            # try to infer suffix from uploaded_file.name if available
            name = getattr(uploaded_file, "name", None) or ""
            suffix = Path(name).suffix or ".pdf"
            temp_path = _save_bytes_to_tempfile(data, suffix=suffix)
            temp_file_to_cleanup = temp_path
            p = temp_path
        elif hasattr(path_or_file, "read") and not isinstance(path_or_file, (str, Path)):
            # Generic file-like object with read()
            data = path_or_file.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
            # infer suffix as above
            header = bytes(data[:8])
            suffix = ".pdf" if header.startswith(b"%PDF") else ".png"
            temp_path = _save_bytes_to_tempfile(data, suffix=suffix)
            temp_file_to_cleanup = temp_path
            p = temp_path
        else:
            # Assume path-like (str or Path)
            if isinstance(path_or_file, Path):
                path_str = str(path_or_file)
            else:
                path_str = str(path_or_file)
            # Normalize and try to resolve to an existing path when possible
            try:
                p = _normalize_and_resolve_path(path_str)
            except Exception:
                p = Path(path_str.replace("\\", "/"))
            # At this point p may or may not exist; check
            if not p.exists():
                raise FileNotFoundError(f"Input not found: {path_or_file} (cwd: {Path.cwd()})")
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to prepare input for extraction: {e}")

    # At this point p is a Path to a file on disk
    suffix = p.suffix.lower()
    logger.debug("Processing file at path: %s (suffix=%s)", p, suffix)

    # PDF handling
    if suffix == ".pdf":
        # Try native extraction with PyMuPDF
        native_text = ""
        if fitz is not None:
            try:
                with fitz.open(str(p)) as doc:
                    parts: List[str] = []
                    for page in doc:
                        try:
                            parts.append(page.get_text("text") or "")
                        except Exception as e:
                            logger.debug("fitz failed to get_text for a page: %s", e)
                            parts.append("")
                    native_text = "\n".join(parts).strip()
            except Exception as e:
                logger.debug("PyMuPDF failed to open or extract text: %s", e)
                native_text = ""
        else:
            logger.debug("PyMuPDF (fitz) is not installed; skipping native text extraction")

        if _is_text_meaningful(native_text):
            logger.debug("Returning native PDF text (length=%d)", len(native_text))
            return native_text

        # If native text not meaningful, fallback to rasterize pages and OCR them
        logger.debug("Native text not meaningful; starting rasterization + OCR fallback")
        try:
            images = _rasterize_pdf_to_pil_images(p, dpi=300)
        except Exception as e:
            raise RuntimeError(f"Failed to rasterize PDF for OCR: {e}")

        ocr_text, mean_conf = _ocr_from_pil_images(images)
        if not _is_text_meaningful(ocr_text):
            raise ValueError("No meaningful text extracted after OCR fallback")
        if mean_conf is not None and mean_conf < MIN_OCR_CONFIDENCE:
            raise ValueError(f"Low OCR confidence: {mean_conf:.4f}")
        logger.debug("Returning OCR text from rasterized PDF (length=%d, conf=%s)", len(ocr_text), mean_conf)
        return ocr_text

    # Image handling
    else:
        if Image is None:
            raise RuntimeError("Pillow is required to process image files but is not installed")

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
        logger.debug("Returning OCR text from image (length=%d, conf=%s)", len(ocr_text), mean_conf)
        return ocr_text
    # unreachable


if __name__ == "__main__":
    # Example usage for local debugging. Update the path below to a local file for testing.
    test_path = Path.cwd() / "data" / "raw" / "sample_invoice.pdf"
    if not test_path.exists():
        # Try a Windows-like sample string if that helps debug path normalization
        test_path = Path(str(test_path).replace("/", "\\"))
    try:
        text = extract_text(test_path)
        print("=== Extracted text (first 1000 chars) ===")
        print(text[:1000])
    except Exception as exc:
        logger.exception("Extraction failed: %s", exc)
        sys.exit(1)
