"""
    04 llm cleaning

llm_cleaning.py 

This file defines the high-level structure for the LLM cleaning pipeline.
Each function is currently empty except for environment setup, which now initializes the Gemini API.

Main entrypoint: llm_cleaner(gemini_api_key, invoice_df, line_items_df)
"""

import os
import re
import json
import pandas as pd
import google.generativeai as genai
from typing import Tuple, Dict, Any
import mimetypes

MODEL_NAME = "gemini-2.5-flash"


def llm_cleaner(gemini_api_key: str, invoice_df: pd.DataFrame, line_items_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Main cleaning pipeline entrypoint."""

    # Step 1: set up environment (API key)
    _setup_environment(gemini_api_key)

    # Step 2: pass dataframes to text method to get JSON (for now empty)
    result_json = _text_cleaning(invoice_df, line_items_df)

    # Step 3: check JSON for missing/invalid data
    visual_needed = _checker(result_json)

    # Step 4: if visual check is needed, call visual method
    if visual_needed:
        result_json = _visual_check(result_json)

    # Step 5: convert JSON back to DataFrames
    updated_invoice_df, updated_line_items_df = _json_to_dataframes(result_json)

    # Step 6: return cleaned DataFrames and whether visual was used
    return updated_invoice_df, updated_line_items_df, visual_needed


def _setup_environment(gemini_api_key: str) -> None:
    """Sets Gemini API key in environment and initializes Gemini client."""
    import os
    os.environ["GEMINI_API_KEY"] = gemini_api_key

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing GEMINI_API_KEY. Please set it as an environment variable.")

    # Configure Gemini client
    genai.configure(api_key=api_key)


def _text_cleaning(invoice_df: pd.DataFrame, line_items_df: pd.DataFrame) -> Dict[str, Any]:
    """Sends invoice and line item data for LLM-based cleaning (returns JSON)."""

    # Extract clean text directly from the invoice_df (no CSV loading)
    if "cleaned_text" not in invoice_df.columns:
        raise ValueError("invoice_df must contain a 'cleaedn_text' column.")

    # Use first cleaned_text for now (assuming one main invoice per df)
    cleaned_text = invoice_df.iloc[0]["cleaned_text"]

    # Convert invoice and line items to dicts
    invoice_data = invoice_df.drop(columns=["cleaedn_text"], errors="ignore").to_dict(orient="records")
    lineitems_data = line_items_df.to_dict(orient="records")

    # Send to Gemini for extraction
    result_json = extract_invoice_json(cleaned_text, invoice_data, lineitems_data)
    return result_json


def _clean_json_output(raw_text: str) -> str:
    """
    Cleans Gemini output by removing markdown code fences (```json, ```),
    stripping whitespace, and ensuring only valid JSON remains.
    """
    if not raw_text:
        raise ValueError("Model returned empty response.")

    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    return cleaned


def extract_invoice_json(cleaned_text: str, invoice_data: list, lineitems_data: list) -> dict:
    """
    Send cleaned invoice text + regex-extracted data to Gemini and return structured JSON.
    """
    prompt = f"""
You are an expert AI that extracts structured invoice data from semi-structured text.

Rules:
1. Output ONLY valid JSON, no explanations.
2. Do NOT wrap JSON in markdown code fences (no ```json or ```).
3. Each line item may contain a description, size, and brand together — separate them correctly.
4. Make sure description doesn't include size or company/brand name. Size and brand should be separate fields.
5. Include a confidence score between 0 and 1 based on extraction reliability.
6. If a field is missing, use null.
7. Ensure totals match quantity * unit_price if possible, otherwise keep the total from the invoice text.
8. Do not include fuel surcharges or taxes as regular line items.
9. The date in source is in format MM-DD-YYYY you have to convert it to DD-MM-YYYY.
10. Use the following schema strictly:

{{
  "llm_invoices": {{
    "file_path": str,
    "vendor": str,
    "date": str,
    "total": float,
    "invoice_number": str
  }},
  "llm_lineitems": [
    {{
      "invoice_number": str,
      "description": str,
      "quantity": float or null,
      "unit_price": float or null,
      "total": float or null
    }}
  ],
  "confidence": float
}}

Cleaned text:
{cleaned_text}

Regex invoice data:
{json.dumps(invoice_data, indent=2)}

Regex line items data:
{json.dumps(lineitems_data, indent=2)}
"""

    # --- Call Gemini model ---
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    raw_text = response.text.strip() if response and response.text else ""

    # --- Clean & parse JSON ---
    cleaned = _clean_json_output(raw_text)

    try:
        parsed_json = json.loads(cleaned)
        return parsed_json
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON even after cleaning. Error: {e}\nRaw output:\n{raw_text}"
        )


import pandas as pd

CONFIDENCE_THRESHOLD = 0.5

def _checker(result_json: Dict[str, Any]) -> bool:
    """
    Checks Gemini text-based extraction for structural issues or low confidence.
    Returns True if visual verification is needed.
    """
    if not result_json:
        return True

    # --- Confidence check ---
    confidence = result_json.get("confidence", 0)
    if confidence is None or confidence < CONFIDENCE_THRESHOLD:
        return True

    invoice = result_json.get("llm_invoices", {})
    lineitems = result_json.get("llm_lineitems", [])

    # --- Invoice sanity check ---
    required_invoice_fields = ["file_path", "vendor", "date", "total", "invoice_number"]
    for field in required_invoice_fields:
        val = invoice.get(field)
        if val in (None, "", "null"):
            return True  # missing critical info

    # --- Line item checks ---
    if not isinstance(lineitems, list) or len(lineitems) == 0:
        return True
    for li in lineitems:
        # description at least must exist
        if not li.get("description"):
            return True
        # sanity check on quantity/price
        q, p, t = li.get("quantity"), li.get("unit_price"), li.get("total")
        if q is not None and p is not None and t is not None:
            calc = round(q * p, 2)
            if abs(calc - t) > 0.05:  # mismatch
                return True

    # passed all checks
    return False


def _json_to_dataframes(result_json: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts Gemini JSON into two DataFrames: invoices_df and lineitems_df.
    Each is normalized and suitable for merging back into the pipeline.
    """
    if not result_json:
        raise ValueError("Empty result_json passed to _json_to_dataframes.")

    invoice_part = result_json.get("llm_invoices", {})
    lineitems_part = result_json.get("llm_lineitems", [])

    invoice_df = pd.DataFrame([{
        "file_path": invoice_part.get("file_path"),
        "vendor": invoice_part.get("vendor"),
        "date": invoice_part.get("date"),
        "total": invoice_part.get("total"),
        "invoice_number": invoice_part.get("invoice_number"),
        "confidence": result_json.get("confidence", None)
    }])

    # Normalize line items
    line_items_records = []
    for li in lineitems_part:
        line_items_records.append({
            "invoice_number": li.get("invoice_number") or invoice_part.get("invoice_number"),
            "description": li.get("description"),
            "quantity": li.get("quantity"),
            "unit_price": li.get("unit_price"),
            "total": li.get("total")
        })

    line_items_df = pd.DataFrame(line_items_records)
    return invoice_df, line_items_df


def _visual_check(result_json: Dict[str, Any]) -> Dict[str, Any]:
    """Handles visual verification using Gemini (returns corrected JSON)."""
    corrected_json = visual_extract_invoice_json(result_json)
    return corrected_json


def visual_extract_invoice_json(existing_json: dict) -> dict:
    """
    Takes the JSON output from extract_invoice_json() and the invoice image/PDF (from file_path).
    Uses Gemini multimodal vision model to correct missing or incorrect fields.

    Args:
        existing_json (dict): JSON from extract_invoice_json().

    Returns:
        dict: Corrected JSON, same schema.
    """
    # --- Validate file path from JSON ---
    file_path = (
        existing_json.get("llm_invoices", {}).get("file_path")
        if existing_json.get("llm_invoices")
        else None
    )
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found or invalid: {file_path}")

    # --- Load file for Gemini ---
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "image/jpeg"  # fallback default
    file_obj = genai.upload_file(file_path, mime_type=mime_type)

    # --- Construct correction prompt ---
    prompt = f"""
You are an expert invoice extraction AI.
You are given a JSON that was generated from an OCR process. 
The JSON may have missing fields or incorrect values.

Your task:
1. Use the provided invoice image/PDF to **verify and correct** all data.
2. Make sure **no field is null or missing** if the information is visible in the document.
3. Keep the same JSON schema.
4. If a field is not visible at all, use null, not a guess.
5. In line items, the "description" field may incorrectly include product size or brand name.
   - Correct format: description / size / brand
   - Only the **first part (product name)** should go in "description".
   - The rest (size, brand) should NOT appear in the "description" field.
   - Example corrections:
       * "flour power 50 lb graincraft" → description="flour power", size="50 lb", brand="graincraft"
       * "jalapeno sliced 6#10 savor" → description="jalapeno sliced", size="6#10", brand="savor"
   - For this schema, **only keep description, quantity, unit_price, total**.
     (size and brand need not appear unless your schema includes them explicitly.)
6. Ensure totals = quantity × unit_price wherever possible.
7. Recalculate confidence between 0 and 1 based on correction reliability.
8. The date in source is in format MM-DD-YYYY you have to convert it to DD-MM-YYYY.
9. Output **only valid JSON** (no markdown or commentary).

Here is the current JSON to correct:
{json.dumps(existing_json, indent=2)}
"""

    # --- Call Gemini multimodal model ---
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content([prompt, file_obj])
    raw_text = response.text.strip() if response and response.text else ""

    # --- Clean & parse JSON ---
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()

    try:
        corrected_json = json.loads(cleaned)
        return corrected_json
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Gemini did not return valid JSON. Error: {e}\nRaw output:\n{raw_text}"
        )

# Example usage
if __name__ == "__main__":
    from regex_lineitems import ExtractRawLineItems

    file_path = r".\data\raw\train\Copy of ARPFIINVOEBTCHLASER (1).pdf"

    # Extract raw invoice and line items from file
    invoices_df, lineitems_df = ExtractRawLineItems(file_path)

    # Run the LLM cleaner
    gemini_api_key = "my-key"
    cleaned_invoices_df, cleaned_lineitems_df, visual_used = llm_cleaner(
        gemini_api_key, invoices_df, lineitems_df
    )

    print("Cleaned Invoice DataFrame:")
    print(cleaned_invoices_df)
    print("\nCleaned Line Items DataFrame:")
    print(cleaned_lineitems_df)
    print(f"\nVisual model used? {visual_used}")
