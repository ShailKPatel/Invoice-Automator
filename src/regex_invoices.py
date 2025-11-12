""""
02 - regex invoices
"""
import re
import pandas as pd
from typing import Tuple, List, Optional
from pathlib import Path

# Import extract_text from the other module
from src.extract_data import extract_text  

# --- Regex definitions ---
_PHONE_RE = re.compile(r"(?:\b(?:PH|PHONE|TEL|FAX|ORDER\s*DESK)\b[:\s]*)?(?:\+?\d{1,2}[\s-]?)?(?:\(?\d{2,3}\)?[\s-]?\d{3}[\s-]?\d{4})", re.I)
_ESCAPE_SEQ_RE = re.compile(r"\\[nrt]")
_HEX_ESC_RE = re.compile(r"\\x[0-9A-Fa-f]{2}")
_BOILERPLATE_RE = re.compile(r"\b(INVOICE|CUSTOMER\s*COPY|ACCOUNTING\s*OFFICE|ORDER\s*DESK|THANK YOU FOR YOUR BUSINESS)\b", re.I)
_MULTISPACE_RE = re.compile(r"\s{2,}")
_WEIGHT_RE = re.compile(r"\b\d+(?:[\s\.,]\d+)*(?:\s*(?:LB|LBS|KG|KGS))\b", re.I)
_DATE_RE = re.compile(r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/([12]\d{3})\b")
_INVOICE_NEAR_RE = re.compile(r"(?:INVOICE[:\s#]*)\s*(\d{6,8})", re.I)
_ID_FALLBACK_RE = re.compile(r"\b(\d{6,8})\b")
_TOTAL_RE = re.compile(r"(?:INVOICE\s*TOTAL|TOTAL[:\s]*\$?)\s*\$?\s*([0-9,]+\.\d{2})", re.I)
_CURRENCY_RE = re.compile(r"\$\s*([0-9]+\.[0-9]{2})")
_LINEITEM_RE = re.compile(
    r"(?P<product_id>\b\d{5,6}\b)\s+(?P<qty>\d+(?:\.\d+)?)\s+CS\s+(?P<desc>[A-Z0-9\-\+\*/\s]{3,80}?)\s+(?P<size>\d+\s*LB)\s+(?P<line_total>\d+\.\d{2})",
    re.I
)


# --- Helpers ---
def clean_text_basic(text: str) -> str:
    text = _ESCAPE_SEQ_RE.sub(" ", text or "")
    text = _HEX_ESC_RE.sub(" ", text)
    text = _WEIGHT_RE.sub(" ", text)
    text = _PHONE_RE.sub(" ", text)
    text = _BOILERPLATE_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text)
    return text.strip().lower()


def extract_vendor(cleaned_text: str) -> Optional[str]:
    if not cleaned_text:
        return None
    m = re.search(r"w{2,3}[\s\-_\.]*([a-z0-9\-]+)[\s\-_\.]+(?:com|net|org|co)", cleaned_text, re.I)
    if m:
        return m.group(1)
    m = re.search(r"[a-z0-9._%+-]+@([a-z0-9\-]+)[\s\._\-]+(?:com|net|org|co)", cleaned_text, re.I)
    if m:
        return m.group(1)
    for t in re.split(r"[|/\n\r]+", cleaned_text):
        t = t.strip()
        if t and re.search(r"[a-z]", t) and len(t.split()) <= 6:
            return t
    return None


def extract_invoice_fields(clean_text: str) -> Tuple[dict, List[dict]]:
    inv = {"invoice_number": None, "vendor": None, "date": None, "total": None}

    m = _INVOICE_NEAR_RE.search(clean_text)
    if m:
        inv["invoice_number"] = m.group(1)
    else:
        m2 = _ID_FALLBACK_RE.search(clean_text)
        if m2:
            inv["invoice_number"] = m2.group(1)

    dates = _DATE_RE.findall(clean_text)
    if dates:
        d = dates[-1]
        inv["date"] = f"{int(d[0]):02d}/{int(d[1]):02d}/{d[2]}"

    m_total = _TOTAL_RE.search(clean_text) or _CURRENCY_RE.search(clean_text)
    if m_total:
        inv["total"] = m_total.group(1).replace(",", "")

    inv["vendor"] = extract_vendor(clean_text)

    line_items = []
    for match in _LINEITEM_RE.finditer(clean_text):
        desc = re.sub(r"\s{2,}", " ", match.group("desc").strip())
        line_items.append({
            "linked_to_invoice": inv["invoice_number"],
            "description": desc,
            "quantity": match.group("qty"),
            "unit_price": None,
            "total": match.group("line_total"),
        })

    return inv, line_items


# --- MAIN FUNCTION ---
def process_raw_data(file_path: str) -> pd.DataFrame:
    """
    Process a single raw invoice file path:
      1. Extract clean text via extract_text()
      2. Apply regex rules
      3. Return a single-row DataFrame with fields and cleaned text
    """
    clean_text = extract_text(file_path)
    cleaned = clean_text_basic(clean_text)
    invoice_data, _ = extract_invoice_fields(cleaned)

    df = pd.DataFrame([{
        "file_path": str(Path(file_path).resolve()),
        "invoice_number": invoice_data.get("invoice_number"),
        "vendor": invoice_data.get("vendor"),
        "date": invoice_data.get("date"),
        "total": invoice_data.get("total"),
        "cleaned_text": cleaned
    }])

    return df


# --- Example usage ---
if __name__ == "__main__":
    test_path = r".\data\raw\train\Copy of ARPFIINVOEBTCHLASER (1).pdf"
    df = process_raw_data(test_path)
    print(df)
