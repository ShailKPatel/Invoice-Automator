import glob
import streamlit as st
import os
import shutil
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
import re
import wordninja
from typing import Tuple


# Suppress warnings globally
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Import from src
from src.regex_lineitems import ExtractRawLineItems
from src.llm_cleaning import llm_cleaner

# -------------------------------
# Vendor Normalization & Date Conversion
# -------------------------------

# Global runtime memory for vendors
known_vendors = set()

# put near top of module
import os
import re
import shutil
import unicodedata
from pathlib import Path
from datetime import datetime
import pandas as pd
import wordninja
from typing import Iterable, Optional

# ensure this exists somewhere in module scope if you rely on it
# known_vendors = set()  # uncomment and populate if needed

# ---------- normalization helpers ----------
def _strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))

def _safe_filename(name: str) -> str:
    # remove path components and dangerous characters, keep extension if present
    base = os.path.basename(name)
    # replace path-separators and control characters
    base = base.replace(os.sep, "_").replace("/", "_")
    # allow letters, digits, dot, dash, underscore, space
    base = re.sub(r"[^0-9A-Za-z\.\-\_\s]", "_", base)
    return base

# ---------- vendor normalization ----------
def normalize_vendor_name(vendor: str, known_vendors_list: Optional[Iterable[str]] = None):
    """
    Normalize and learn vendor names across OSes. Returns a cleaned title-cased vendor string.
    known_vendors_list can be a set/list to match against and to add new learned names.
    """
    if not vendor:
        return ""

    if known_vendors_list is None:
        try:
            known_vendors_list = known_vendors  # fallback to module-level variable
        except NameError:
            known_vendors_list = set()

    # unicode normalize and strip surrounding whitespace
    vendor_clean = _strip_accents(str(vendor)).strip()

    # keep spaces and word characters so wordninja has something sensible to split
    vendor_alnum_space = re.sub(r"[^\w\s]", "", vendor_clean, flags=re.UNICODE)
    vendor_lower = vendor_alnum_space.lower().strip()

    # remove common suffixes as whole words (not by raw endswith only)
    suffixes = {"inc", "incorporated", "ltd", "llc", "company", "corp", "co"}
    parts = [p for p in vendor_lower.split() if p and p not in suffixes]
    vendor_lower = " ".join(parts).strip()

    # Step 3 – match known vendors (normalize comparison)
    for known in list(known_vendors_list):
        norm_known = re.sub(r"\s+", "", _strip_accents(str(known)).lower())
        norm_candidate = re.sub(r"\s+", "", vendor_lower)
        if norm_known and norm_known in norm_candidate:
            return str(known)

    # Step 4 – split concatenated words with wordninja and title case
    words = wordninja.split(vendor_lower.replace("_", ""))
    cleaned = " ".join(words).strip().title()

    # Step 5 – store learned vendor name if set-like
    if isinstance(known_vendors_list, set):
        known_vendors_list.add(cleaned)
    elif hasattr(known_vendors_list, "append"):
        known_vendors_list.append(cleaned)

    return cleaned

# ---------- date conversion ----------
def convert_date_format(date_str):
    """Convert date to DD-MM-YYYY format if possible, robust across locales."""
    if pd.isna(date_str) or not str(date_str).strip():
        return ""
    s = str(date_str).strip()
    known_formats = ("%d-%m-%Y", "%Y-%m-%d", "%m-%d-%Y", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y")
    for fmt in known_formats:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%d-%m-%Y")
        except ValueError:
            continue
    # fallback: let pandas try, prefer dayfirst to handle DD/MM/YYYY like inputs
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.notna(dt):
            return dt.strftime("%d-%m-%Y")
    except Exception:
        pass
    return s

# ---------- temp folder helpers (Path-based) ----------
def get_root_temp_folder() -> Path:
    """
    Returns Path to repo-root/invoice_temp_storage.
    Uses Path(__file__).resolve().parents[1] which is reliable for typical package layout.
    """
    root_dir = Path(__file__).resolve().parents[1]
    temp_folder = root_dir / "invoice_temp_storage"
    temp_folder.mkdir(parents=True, exist_ok=True)
    return temp_folder

def clean_temp_storage(temp_folder: (str | Path)):
    temp_folder = Path(temp_folder)
    processed_folder = temp_folder / "processed"
    regex_folder = temp_folder / "regex"
    llm_folder = temp_folder / "llm"

    for folder in (processed_folder, regex_folder, llm_folder):
        folder.mkdir(parents=True, exist_ok=True)
        for item in folder.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
            except Exception:
                # keep going, do not crash cleanup
                pass

    # Create blank LLM CSV templates
    llm_invoices_path = llm_folder / "llm_invoices.csv"
    llm_lineitems_path = llm_folder / "llm_lineitems.csv"

    invoices_cols = ["file_path", "invoice_id", "vendor", "date", "total", "invoice_number", "visual_used"]
    lineitems_cols = ["file_path", "invoice_id", "description", "quantity", "unit_price", "total"]

    pd.DataFrame(columns=invoices_cols).to_csv(llm_invoices_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=lineitems_cols).to_csv(llm_lineitems_path, index=False, encoding="utf-8-sig")

    # Create blank REGEX CSV template
    regex_invoices_path = regex_folder / "regex_invoices.csv"
    regex_lineitems_path = regex_folder / "regex_lineitems.csv"

    regex_invoices_cols = ["file_path", "vendor", "date", "total", "invoice_number", "cleaned_text"]
    regex_lineitems_cols = ["file_path", "invoice_number", "description", "quantity", "unit_price", "total"]

    pd.DataFrame(columns=regex_invoices_cols).to_csv(regex_invoices_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(columns=regex_lineitems_cols).to_csv(regex_lineitems_path, index=False, encoding="utf-8-sig")

    return processed_folder, regex_folder, llm_folder

def save_uploaded_files(uploaded_files, folder_path):
    """
    Save uploaded files (Streamlit UploadedFile or similar) safely and return list of Path objects.
    """
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    saved_files = []
    for uploaded_file in uploaded_files:
        safe_name = _safe_filename(getattr(uploaded_file, "name", str(uploaded_file)))
        dest = folder / safe_name
        # Streamlit uploaded file may be BytesIO-like with .read(); support both
        content = None
        try:
            content = uploaded_file.read()
        except Exception:
            try:
                content = uploaded_file.getbuffer()
            except Exception:
                pass
        if content is None:
            # fallback: try to convert to bytes
            content = bytes(uploaded_file)
        with open(dest, "wb") as f:
            f.write(content)
        saved_files.append(dest)
    return saved_files

def process_files_to_processed_folder(temp_folder, processed_folder):
    temp_folder = Path(temp_folder)
    processed_folder = Path(processed_folder)
    processed_folder.mkdir(parents=True, exist_ok=True)

    for p in temp_folder.iterdir():
        if p.is_dir():
            continue
        suffix = p.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg"}:
            shutil.move(str(p), str(processed_folder / p.name))
        elif suffix == ".pdf":
            reader = PdfReader(str(p))
            num_pages = len(reader.pages)
            if num_pages <= 1:
                shutil.move(str(p), str(processed_folder / p.name))
            else:
                base_name = p.stem
                for i, page in enumerate(reader.pages, start=1):
                    writer = PdfWriter()
                    writer.add_page(page)
                    new_filename = f"{base_name}-page{i}.pdf"
                    new_path = processed_folder / new_filename
                    with open(new_path, "wb") as output_file:
                        writer.write(output_file)
                p.unlink()  # remove original multi-page file

    return processed_folder

def run_regex_extraction(processed_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process every file under `processed_folder` (recursively). For each file call
    ExtractRawLineItems(file_path) and collect returned 'invoice' and 'line_items'
    into two CSVs and two DataFrames.

    Returns:
        (inv_df, li_df)
    """

    invoices_cols = ["file_path", "vendor", "date", "total", "invoice_number", "cleaned_text"]
    lineitems_cols = ["file_path", "invoice_number", "description", "quantity", "unit_price", "total"]

    invoices = []
    lineitems = []

    # Find all files recursively
    file_paths = [fp for fp in glob.glob(os.path.join(processed_folder, "**/*.*"), recursive=True) if os.path.isfile(fp)]


def run_regex_extraction(processed_folder: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process every file under `processed_folder` (recursively). For each file call
    ExtractRawLineItems(file_path) and collect returned 'invoice' and 'line_items'
    into two CSVs and two DataFrames.

    Returns:
        (inv_df, li_df)
    """

    invoices_cols = ["file_path", "vendor", "date", "total", "invoice_number", "cleaned_text"]
    lineitems_cols = ["file_path", "invoice_number", "description", "quantity", "unit_price", "total"]

    invoices = []
    lineitems = []

    # Find all files recursively
    file_paths = [
        fp for fp in glob.glob(os.path.join(processed_folder, "**/*.*"), recursive=True)
        if os.path.isfile(fp)
    ]

    for file_path in file_paths:
        try:
            invoices_df, lineitems_df = ExtractRawLineItems(file_path)
        except Exception as e:
            print(f"⚠️ Skipping {file_path} due to error: {e}")
            continue

        # Validate both DataFrames
        if not isinstance(invoices_df, pd.DataFrame) or invoices_df.empty:
            print(f"⚠️ No invoice data found for {file_path}")
            continue
        if not isinstance(lineitems_df, pd.DataFrame) or lineitems_df.empty:
            print(f"⚠️ No line items found for {file_path}")
            continue

        # Build invoice records
        for _, inv_row in invoices_df.iterrows():
            invoices.append({
                "file_path": file_path,
                "vendor": str(inv_row.get("vendor", "")),
                "date": str(inv_row.get("date", "")),
                "total": str(inv_row.get("total", "")),
                "invoice_number": str(inv_row.get("invoice_number", "")),
                "cleaned_text": str(inv_row.get("cleaned_text", "")),
            })

        # Build line item records
        for _, li_row in lineitems_df.iterrows():
            lineitems.append({
                "file_path": file_path,
                "invoice_number": str(li_row.get("invoice_number", "")),
                "description": str(li_row.get("description", "")),
                "quantity": str(li_row.get("quantity", "")),
                "unit_price": str(li_row.get("unit_price", "")),
                "total": str(li_row.get("total", "")),
            })

    # Create DataFrames
    inv_df = pd.DataFrame(invoices, columns=invoices_cols) if invoices else pd.DataFrame(columns=invoices_cols)
    li_df = pd.DataFrame(lineitems, columns=lineitems_cols) if lineitems else pd.DataFrame(columns=lineitems_cols)

    # Ensure consistent types (avoid pyarrow ArrowTypeError)
    inv_df = inv_df.astype(str)
    li_df = li_df.astype(str)

    # Hardcoded save paths
    base_dir = os.path.join("invoice_temp_storage", "llm")
    os.makedirs(base_dir, exist_ok=True)

    # invoices_csv_path = os.path.join(base_dir, "llm_invoices.csv")
    # lineitems_csv_path = os.path.join(base_dir, "llm_lineitems.csv")

    # try:
    #     inv_df.to_csv(invoices_csv_path, index=False, encoding="utf-8")
    #     li_df.to_csv(lineitems_csv_path, index=False, encoding="utf-8")
    #     print(f"Data written to:\n  - {invoices_csv_path}\n  - {lineitems_csv_path}")
    # except Exception as e:
    #     print(f"⚠️ Error writing CSV files: {e}")

    print(f"✅ Regex extraction complete. Files processed: {len(file_paths)}")
    print(f"Invoices found: {len(inv_df)} | Line items found: {len(li_df)}")
    return inv_df, li_df

def process_invoices_with_llm(gemini_key, inv_df: pd.DataFrame, li_df: pd.DataFrame, llm_folder: str):
    """
    For each unique file_path in inv_df:
      - collect the invoice row(s) for that file_path (should be unique per instructions)
      - collect matching line items from li_df (line items may have multiple rows per file_path)
      - call llm_cleaner(gemini_api_key, invoices_df, lineitems_df)
      - receive cleaned_invoices_df, cleaned_lineitems_df, visual_used (boolean)
      - add visual_used to cleaned_invoices_df as column 'visual_used'
      - ensure invoice_id exists for both tables; fill missing with 'xyz'
      - append results to persistent CSVs:
            ../invoice_temp_storage/llm/llm_invoices.csv
            ../invoice_temp_storage/llm/llm_lineitems.csv
    Returns:
        (llm_invoices_df, llm_lineitems_df)
    """

    # Normalize folder paths
    llm_folder = Path(llm_folder)
    llm_folder.mkdir(parents=True, exist_ok=True)

    llm_inv_path = llm_folder / "llm_invoices.csv"
    llm_li_path = llm_folder / "llm_lineitems.csv"

    # Safety: handle None input
    inv_df = inv_df.copy() if inv_df is not None else pd.DataFrame(columns=[])
    li_df = li_df.copy() if li_df is not None else pd.DataFrame(columns=[])

    # Prepare storage DataFrames (load existing if present)
    if llm_inv_path.exists():
        all_inv_df = pd.read_csv(llm_inv_path)
    else:
        all_inv_df = pd.DataFrame()

    if llm_li_path.exists():
        all_li_df = pd.read_csv(llm_li_path)
    else:
        all_li_df = pd.DataFrame()

    # Static counter for unique invoice IDs
    invoice_counter = 1

    # Collect all unique file paths
    unique_file_paths = inv_df["file_path"].unique().tolist() if "file_path" in inv_df.columns else []

    for file_path in unique_file_paths:
        try:
            # Get subset for this file
            temp_inv_df = inv_df[inv_df["file_path"] == file_path].copy()
            temp_li_df = li_df[li_df["file_path"] == file_path].copy()

            if temp_inv_df.empty:
                continue

            # --- call LLM cleaner ---
            cleaned_invoices_df, cleaned_lineitems_df, visual_used = llm_cleaner(
                gemini_key, temp_inv_df, temp_li_df
            )

            # Defensive defaults
            if cleaned_invoices_df is None:
                cleaned_invoices_df = pd.DataFrame(columns=temp_inv_df.columns)
            if cleaned_lineitems_df is None:
                cleaned_lineitems_df = pd.DataFrame(columns=temp_li_df.columns)

            # --- Invoices post-processing ---
            cleaned_invoices_df = cleaned_invoices_df.copy()
            cleaned_invoices_df["file_path"] = file_path

            # Normalize vendor names if function exists
            if "vendor" in cleaned_invoices_df.columns:
                cleaned_invoices_df["vendor"] = cleaned_invoices_df["vendor"].apply(
                    lambda v: normalize_vendor_name(v) if pd.notnull(v) else v
                )

            cleaned_invoices_df["visual_used"] = bool(visual_used)

            # Generate a unique invoice_id for this file_path
            unique_invoice_id = f"inv_{invoice_counter:05d}"
            invoice_counter += 1

            # Overwrite or create invoice_id with the unique value
            cleaned_invoices_df["invoice_id"] = unique_invoice_id

            # Remove cleaned_text column if present
            if "cleaned_text" in cleaned_invoices_df.columns:
                cleaned_invoices_df = cleaned_invoices_df.drop(columns=["cleaned_text"])

            # Reorder & guarantee all expected columns (without cleaned_text)
            inv_out_cols = [
                "file_path", "invoice_id", "vendor", "date", "total",
                "invoice_number", "visual_used"
            ]
            for c in inv_out_cols:
                if c not in cleaned_invoices_df.columns:
                    cleaned_invoices_df[c] = ""

            cleaned_invoices_df = cleaned_invoices_df[inv_out_cols]

            # --- Line items post-processing ---
            cleaned_lineitems_df = cleaned_lineitems_df.copy()
            cleaned_lineitems_df["file_path"] = file_path

            # Ensure invoice_id (overwrite with same unique id for this file_path)
            if "invoice_id" not in cleaned_lineitems_df.columns:
                cleaned_lineitems_df["invoice_id"] = unique_invoice_id
            else:
                cleaned_lineitems_df["invoice_id"] = unique_invoice_id

            # --- Append to global CSVs ---
            all_inv_df = pd.concat([all_inv_df, cleaned_invoices_df], ignore_index=True)
            all_li_df = pd.concat([all_li_df, cleaned_lineitems_df], ignore_index=True)

            # Persist
            all_inv_df.to_csv(llm_inv_path, index=False)
            all_li_df.to_csv(llm_li_path, index=False)

        except Exception as e:
            print(f"⚠️ Error processing {file_path}: {e}")
            continue

    # Return final DataFrames on disk
    return all_inv_df, all_li_df



# -------------------------------
# Main Streamlit Page
# -------------------------------
temp_folder = get_root_temp_folder()
processed_folder, regex_folder, llm_folder = clean_temp_storage(temp_folder)
inv_path = os.path.join(regex_folder, "regex_invoices.csv")
li_path = os.path.join(regex_folder, "regex_lineitems.csv")

st.title("Invoice Automation")
st.markdown("Upload up to **15 invoice files** (PDF or image). The app extracts text using OCR, parses invoices with Regex, and refines results using Gemini LLM.")
st.divider()

uploaded_files = st.file_uploader("Upload invoice files:", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    if 1 <= len(uploaded_files) <= 15:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded.")
        save_uploaded_files(uploaded_files, temp_folder)
        process_files_to_processed_folder(temp_folder, processed_folder)

        gemini_key = st.text_input("Enter your Google Gemini API key:", type="password")
        confirm_api = st.checkbox("I confirm that I have entered a valid API key and accept to proceed.")

        if confirm_api and st.button("Start Extraction"):


            st.info("Running OCR and Regex extraction...")
            inv_df, li_df = run_regex_extraction(processed_folder)
            st.success("✅ OCR and Regex extraction completed.")

            st.markdown("#### Regex Invoices")
            st.dataframe(inv_df, hide_index=True)
            st.markdown("#### Regex Line Items")
            st.dataframe(li_df, hide_index=True)

            st.info("Running LLM Invoice Processor...")
            try:
                llm_invoices_df, llm_lineitems_df = process_invoices_with_llm(gemini_key, inv_df, li_df, llm_folder)
            except Exception as e:
                if "API key not valid" in str(e) or "API_KEY_INVALID" in str(e):
                    st.error("❌ Invalid Gemini API Key. Please enter a valid key to continue.")
                    st.stop()
                else:
                    st.error(f"❌ LLM processing failed: {e}")
                    st.stop()

            if llm_invoices_df is None or llm_lineitems_df is None or llm_invoices_df.empty:
                st.error("❌ LLM processing returned no data. Please check your API key or input files.")
                st.stop()

            # --- Normalize Vendor & Date Fields ---
            if "date" in llm_invoices_df.columns:
                llm_invoices_df["date"] = llm_invoices_df["date"].apply(convert_date_format)
            if "vendor" in llm_invoices_df.columns:
                llm_invoices_df["vendor"] = llm_invoices_df["vendor"].apply(normalize_vendor_name)

            st.success("✅ LLM processing completed successfully.")
            llm_invoices_df["file_path"] = llm_invoices_df["file_path"].apply(os.path.basename)
            llm_lineitems_df["file_path"] = llm_lineitems_df["file_path"].apply(os.path.basename)

            st.markdown("#### LLM Invoices")
            st.dataframe(llm_invoices_df, hide_index=True)
            st.download_button("Download LLM Invoices CSV", data=llm_invoices_df.to_csv(index=False).encode("utf-8-sig"), file_name="llm_invoices.csv")
            st.markdown("#### LLM Line Items")
            st.dataframe(llm_lineitems_df, hide_index=True)
            st.download_button("Download LLM Line Items CSV", data=llm_lineitems_df.to_csv(index=False).encode("utf-8-sig"), file_name="llm_lineitems.csv")

            # -------------------------------
            # Visualization Section
            # -------------------------------
            st.divider()
            st.subheader("Business Insights Dashboard")

            if not llm_invoices_df.empty:
                llm_invoices_df["total"] = pd.to_numeric(llm_invoices_df["total"], errors="coerce")

                # Convert cleaned date back to datetime for charts
                if "date" in llm_invoices_df.columns:
                    llm_invoices_df["date"] = pd.to_datetime(llm_invoices_df["date"], format="%d-%m-%Y", errors="coerce")

                st.markdown("### Total Spend by Vendor")
                spend_df = (
                    llm_invoices_df.groupby("vendor", as_index=False)["total"]
                    .sum()
                    .sort_values("total", ascending=False)
                )

                if not spend_df.empty:
                    plt.figure(figsize=(8, 4))
                    plt.bar(spend_df["vendor"], spend_df["total"])
                    plt.xticks(rotation=45, ha="right")
                    plt.xlabel("Vendor")
                    plt.ylabel("Total Spend")
                    plt.tight_layout()
                    st.pyplot(plt)


                st.markdown("### Average Invoice Value by Vendor")
                avg_invoice_df = (
                    llm_invoices_df.groupby("vendor", as_index=False)["total"]
                    .mean()
                    .sort_values("total", ascending=False)
                )

                if not avg_invoice_df.empty:
                    plt.figure(figsize=(8, 4))
                    plt.bar(avg_invoice_df["vendor"], avg_invoice_df["total"])
                    plt.xticks(rotation=45, ha="right")
                    plt.xlabel("Vendor")
                    plt.ylabel("Average Invoice Value")
                    plt.tight_layout()
                    st.pyplot(plt)

            if not llm_lineitems_df.empty:
                st.markdown("### Top 10 Purchased Items (by Total Spend)")
                llm_lineitems_df["total"] = pd.to_numeric(llm_lineitems_df["total"], errors="coerce")

                top_items = (
                    llm_lineitems_df.groupby("description", as_index=False)["total"]
                    .sum()
                    .sort_values("total", ascending=False)
                    .head(10)
                )

                if not top_items.empty:
                    plt.figure(figsize=(8, 4))
                    plt.barh(top_items["description"], top_items["total"])
                    plt.gca().invert_yaxis()
                    plt.xlabel("Total Spend")
                    plt.ylabel("Item Description")
                    plt.tight_layout()
                    st.pyplot(plt)
            
            # Pie chart for Visual Usage
            if "visual_use" in llm_lineitems_df.columns:
                st.markdown("### Visual Use Distribution")

                visual_counts = llm_lineitems_df["visual_used"].value_counts(dropna=False)

                if not visual_counts.empty:
                    plt.figure(figsize=(4, 4))
                    plt.pie(
                        visual_counts,
                        labels=visual_counts.index.map({True: "Used", False: "Not Used"}).astype(str),
                        autopct="%1.1f%%",
                        startangle=90,
                    )
                    plt.title("Visual Use Split")
                    plt.tight_layout()
                    st.pyplot(plt)


    else:
        st.error("⚠️ You can upload between 1 and 15 files only.")
else:
    st.info("Please upload invoices to begin.")
