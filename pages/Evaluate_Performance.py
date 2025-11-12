import streamlit as st
import pandas as pd
import io
import os
import re

st.set_page_config(page_title="Evaluate Performance", layout="centered")

# Page title
st.title("Evaluate Performance")

# Section: Answer Key
st.header("Answer Key")
st.write("Upload the ground-truth CSV files for invoices and line items.")
answer_invoice_file = st.file_uploader("Upload answer (ground-truth) invoices CSV", type=["csv"], key="ans_inv")
answer_lineitems_file = st.file_uploader("Upload answer (ground-truth) lineitems CSV", type=["csv"], key="ans_li")

st.markdown("---")

# Section: Submitted/Extracted Output
st.header("Submitted Extraction")
st.write("Upload the extracted/predicted CSV files that you want to evaluate against the answer key.")
submitted_invoice_file = st.file_uploader("Upload submitted invoices CSV", type=["csv"], key="sub_inv")
submitted_lineitems_file = st.file_uploader("Upload submitted lineitems CSV", type=["csv"], key="sub_li")

st.markdown("---")

# Required columns (as specified)
REQUIRED_INVOICE_COLUMNS = ["file_path","invoice_id","vendor","date","total","invoice_number"]
REQUIRED_LINEITEM_COLUMNS = ["file_path","invoice_id","description","quantity","unit_price","total"]


def read_csv_upload(uploaded_file):
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        return None, str(e)


def validate_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    return missing


def filename_extension_key(p):
    """Return the basename (filename.extension) for matching.
    Only strip directory/root folders. Do not split on dashes/underscores or other tokens.
    Examples:
      '/path/to/Copy of ARPFIINVOEBTCHLASER (4)-page1.jpg' -> 'copy of arpfiinvoebtchlaser (4)-page1.jpg'
      'invoice123.pdf' -> 'invoice123.pdf'
    """
    if pd.isna(p):
        return ""
    s = str(p)
    base = os.path.basename(s)
    return base.lower()


def normalize_text(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    return " ".join(s.split()).lower()


if st.button("Run basic validation checks"):
    # Ensure all four files are provided
    if not all([answer_invoice_file, answer_lineitems_file, submitted_invoice_file, submitted_lineitems_file]):
        st.error("All four CSVs must be uploaded: answer invoices, answer lineitems, submitted invoices, submitted lineitems.")
        st.stop()

    # Read CSVs
    ans_inv_df, err = read_csv_upload(answer_invoice_file)
    if err:
        st.error(f"Failed to read answer invoices CSV: {err}")
        st.stop()

    ans_li_df, err = read_csv_upload(answer_lineitems_file)
    if err:
        st.error(f"Failed to read answer lineitems CSV: {err}")
        st.stop()

    sub_inv_df, err = read_csv_upload(submitted_invoice_file)
    if err:
        st.error(f"Failed to read submitted invoices CSV: {err}")
        st.stop()

    sub_li_df, err = read_csv_upload(submitted_lineitems_file)
    if err:
        st.error(f"Failed to read submitted lineitems CSV: {err}")
        st.stop()

    # Validate required columns
    missing = validate_columns(ans_inv_df, REQUIRED_INVOICE_COLUMNS)
    if missing:
        st.error(f"Answer invoices CSV is missing required columns: {missing}")
        st.stop()

    missing = validate_columns(ans_li_df, REQUIRED_LINEITEM_COLUMNS)
    if missing:
        st.error(f"Answer lineitems CSV is missing required columns: {missing}")
        st.stop()

    missing = validate_columns(sub_inv_df, REQUIRED_INVOICE_COLUMNS)
    if missing:
        st.error(f"Submitted invoices CSV is missing required columns: {missing}")
        st.stop()

    missing = validate_columns(sub_li_df, REQUIRED_LINEITEM_COLUMNS)
    if missing:
        st.error(f"Submitted lineitems CSV is missing required columns: {missing}")
        st.stop()

    # Create filename.extension keys for invoices (invoice-level uniqueness)
    ans_inv_df['fname_ext'] = ans_inv_df['file_path'].apply(filename_extension_key)
    sub_inv_df['fname_ext'] = sub_inv_df['file_path'].apply(filename_extension_key)

    if ans_inv_df['fname_ext'].duplicated().any():
        st.error("Answer invoices CSV: filename.extension values must be unique per invoice. Duplicate values found.")
        st.stop()

    if sub_inv_df['fname_ext'].duplicated().any():
        st.error("Submitted invoices CSV: filename.extension values must be unique per invoice. Duplicate values found.")
        st.stop()

    # Check that unique filename.extension count match between answer and submitted invoice CSVs
    ans_paths = set(ans_inv_df['fname_ext'].astype(str).unique())
    sub_paths = set(sub_inv_df['fname_ext'].astype(str).unique())

    if len(ans_paths) != len(sub_paths):
        st.error(f"Number of unique filename.extension in answer invoices ({len(ans_paths)}) does not match submitted invoices ({len(sub_paths)}).")
        st.stop()

    # For lineitems: only validate columns and CSV validity. We'll enforce uniqueness per (filename.extension + description) instead of per filename alone.
    ans_li_df['fname_ext'] = ans_li_df['file_path'].apply(filename_extension_key)
    sub_li_df['fname_ext'] = sub_li_df['file_path'].apply(filename_extension_key)

    # Check that in lineitems, the pair (fname_ext, normalized_description) is unique in the answer key
    ans_li_df['ndesc'] = ans_li_df['description'].apply(normalize_text)
    dup_mask = ans_li_df.duplicated(subset=['fname_ext', 'ndesc'])
    if dup_mask.any():
        st.error("Answer lineitems CSV: duplicate (filename.extension + description) pairs found. Each line item should be uniquely identified by filename and description.")
        st.stop()

    # If reached here, basic validations passed
    st.success("Basic validation checks passed. Required columns present and filename.extension uniqueness / counts are consistent.")
    st.write(f"Number of invoices (unique filename.extension): {len(ans_paths)}")
    st.write(f"Number of lineitem rows (answer): {len(ans_li_df)}")
    st.write(f"Number of lineitem rows (submitted): {len(sub_li_df)}")

    # Show small preview of each dataframe
    with st.expander("Preview: answer invoices"):
        st.dataframe(ans_inv_df.head())
    with st.expander("Preview: answer lineitems"):
        st.dataframe(ans_li_df.head())
    with st.expander("Preview: submitted invoices"):
        st.dataframe(sub_inv_df.head())
    with st.expander("Preview: submitted lineitems"):
        st.dataframe(sub_li_df.head())

    st.info("Now you can run the full evaluation to compute field-level metrics (accuracy, precision, recall, F1). Configure fuzzy matching and numeric tolerance below, then click 'Run full evaluation'.")

# === Evaluation: Field-level matching and metrics ===

st.header("Field-level Evaluation: accuracy, precision, recall, F1")

use_fuzzy_vendor = st.checkbox("Enable fuzzy matching for vendor names (uses string similarity)", value=False)
vendor_threshold = st.slider("Vendor fuzzy match threshold (0-1)", 0.0, 1.0, 0.85, 0.01) if use_fuzzy_vendor else None

numeric_tolerance = st.number_input("Numeric tolerance for totals and prices (absolute)", value=0.01, format="%.4f")

from difflib import SequenceMatcher
from datetime import datetime


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def parse_date_try(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    # try common formats
    fmts = ["%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d %b %Y", "%d %B %Y"]
    for f in fmts:
        try:
            return datetime.strptime(s, f).date()
        except Exception:
            continue
    # last resort: try pandas
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors='coerce')
        if pd.isna(dt):
            return None
        return dt.date()
    except Exception:
        return None


def numeric_equal(a, b, tol):
    try:
        if pd.isna(a) and pd.isna(b):
            return True
        if pd.isna(a) or pd.isna(b):
            return False
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def compute_metrics(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


if st.button("Run full evaluation"):
    # Re-run basic validations to ensure dataframes loaded
    if not all([answer_invoice_file, answer_lineitems_file, submitted_invoice_file, submitted_lineitems_file]):
        st.error("All four CSVs must be uploaded before evaluation.")
        st.stop()

    ans_inv_df, err = read_csv_upload(answer_invoice_file)
    ans_li_df, err2 = read_csv_upload(answer_lineitems_file)
    sub_inv_df, err3 = read_csv_upload(submitted_invoice_file)
    sub_li_df, err4 = read_csv_upload(submitted_lineitems_file)
    if err or err2 or err3 or err4:
        st.error("Error reading one of the CSVs. Run basic validation checks first.")
        st.stop()

    # Create filename.extension keys and set index by them for invoice-level matching
    ans_inv_df['fname_ext'] = ans_inv_df['file_path'].apply(filename_extension_key)
    sub_inv_df['fname_ext'] = sub_inv_df['file_path'].apply(filename_extension_key)

    ans_inv_df.set_index('fname_ext', inplace=True)
    sub_inv_df.set_index('fname_ext', inplace=True)

    invoice_keys = sorted(list(ans_inv_df.index.astype(str)))

    # Fields to evaluate for invoices
    invoice_fields = ["vendor", "date", "total", "invoice_number"]

    # Counters per field
    counters = {f: {"tp": 0, "fp": 0, "fn": 0} for f in invoice_fields}

    for key in invoice_keys:
        # ans row exists by design
        ans_row = ans_inv_df.loc[key]
        if key not in sub_inv_df.index:
            # all fields are FN if submission missing
            for f in invoice_fields:
                if pd.notna(ans_row[f]):
                    counters[f]['fn'] += 1
            continue
        sub_row = sub_inv_df.loc[key]

        # vendor - fuzzy optional
        a = normalize_text(ans_row['vendor'])
        b = normalize_text(sub_row['vendor'])
        match = False
        if use_fuzzy_vendor:
            score = similar(a, b)
            match = score >= vendor_threshold
        else:
            match = (a == b and a != "")
        if match:
            counters['vendor']['tp'] += 1
        else:
            if a != "":
                counters['vendor']['fn'] += 1
            if b != "":
                counters['vendor']['fp'] += 1

        # date - parse
        a_date = parse_date_try(ans_row['date'])
        b_date = parse_date_try(sub_row['date'])
        if a_date is not None and b_date is not None and a_date == b_date:
            counters['date']['tp'] += 1
        else:
            if a_date is not None:
                counters['date']['fn'] += 1
            if b_date is not None:
                counters['date']['fp'] += 1

        # total - numeric tolerance
        if numeric_equal(ans_row['total'], sub_row['total'], numeric_tolerance):
            counters['total']['tp'] += 1
        else:
            if pd.notna(ans_row['total']):
                counters['total']['fn'] += 1
            if pd.notna(sub_row['total']):
                counters['total']['fp'] += 1

        # invoice_number
        a = normalize_text(ans_row['invoice_number'])
        b = normalize_text(sub_row['invoice_number'])
        if a == b and a != "":
            counters['invoice_number']['tp'] += 1
        else:
            if a != "":
                counters['invoice_number']['fn'] += 1
            if b != "":
                counters['invoice_number']['fp'] += 1

    # Compute metrics per invoice field
    st.subheader("Invoice-level metrics")
    inv_metrics = {}
    for f, c in counters.items():
        m = compute_metrics(c['tp'], c['fp'], c['fn'])
        inv_metrics[f] = m
    st.json(inv_metrics)

    # Overall invoice-level averages
    avg_prec = sum(m['precision'] for m in inv_metrics.values()) / len(inv_metrics)
    avg_rec = sum(m['recall'] for m in inv_metrics.values()) / len(inv_metrics)
    avg_f1 = sum(m['f1'] for m in inv_metrics.values()) / len(inv_metrics)
    st.write(f"Invoice-level average Precision: {avg_prec:.3f}, Recall: {avg_rec:.3f}, F1: {avg_f1:.3f}")

    # === Line items evaluation ===
    st.subheader("Line-item level evaluation (description-based matching)")

    # Create filename.extension keys for lineitems and normalized description
    ans_li_df['fname_ext'] = ans_li_df['file_path'].apply(filename_extension_key)
    sub_li_df['fname_ext'] = sub_li_df['file_path'].apply(filename_extension_key)
    ans_li_df['ndesc'] = ans_li_df['description'].apply(normalize_text)
    sub_li_df['ndesc'] = sub_li_df['description'].apply(normalize_text)

    line_fields = ["description", "quantity", "unit_price", "total"]
    lf_counters = {f: {"tp": 0, "fp": 0, "fn": 0} for f in line_fields}

    # For each invoice (by filename.extension), match submitted lineitems to answer lineitems by description (not by order)
    for key in invoice_keys:
        ans_items = ans_li_df[ans_li_df['fname_ext'] == key].copy()
        sub_items = sub_li_df[sub_li_df['fname_ext'] == key].copy()

        # build map of normalized description -> list of rows for answer
        ans_map = {}
        for i, row in ans_items.iterrows():
            ans_map.setdefault(row['ndesc'], []).append(row)

        # Match each submitted item to an answer item using description similarity, not order
        for j, srow in sub_items.iterrows():
            sdesc = srow['ndesc']
            best_desc = None
            best_score = 0.0
            # try exact normalized match first
            if sdesc in ans_map and len(ans_map[sdesc]) > 0:
                best_desc = sdesc
                best_score = 1.0
            else:
                # fuzzy search
                for adesc in list(ans_map.keys()):
                    score = similar(sdesc, adesc)
                    if score > best_score and len(ans_map.get(adesc, [])) > 0:
                        best_score = score
                        best_desc = adesc
            # Decide match using fuzzy threshold 0.85 for descriptions
            if best_score >= 0.85 and best_desc is not None and len(ans_map.get(best_desc, [])) > 0:
                # pop one answer row
                arow = ans_map[best_desc].pop(0)
                # description matched
                lf_counters['description']['tp'] += 1
                # quantity
                if numeric_equal(arow['quantity'], srow['quantity'], numeric_tolerance):
                    lf_counters['quantity']['tp'] += 1
                else:
                    if pd.notna(arow['quantity']):
                        lf_counters['quantity']['fn'] += 1
                    if pd.notna(srow['quantity']):
                        lf_counters['quantity']['fp'] += 1
                # unit_price
                if numeric_equal(arow['unit_price'], srow['unit_price'], numeric_tolerance):
                    lf_counters['unit_price']['tp'] += 1
                else:
                    if pd.notna(arow['unit_price']):
                        lf_counters['unit_price']['fn'] += 1
                    if pd.notna(srow['unit_price']):
                        lf_counters['unit_price']['fp'] += 1
                # total
                if numeric_equal(arow['total'], srow['total'], numeric_tolerance):
                    lf_counters['total']['tp'] += 1
                else:
                    if pd.notna(arow['total']):
                        lf_counters['total']['fn'] += 1
                    if pd.notna(srow['total']):
                        lf_counters['total']['fp'] += 1
            else:
                # no match found for this submitted item
                lf_counters['description']['fp'] += 1
                if pd.notna(srow['quantity']):
                    lf_counters['quantity']['fp'] += 1
                if pd.notna(srow['unit_price']):
                    lf_counters['unit_price']['fp'] += 1
                if pd.notna(srow['total']):
                    lf_counters['total']['fp'] += 1

        # Any remaining items in ans_map are unmatched -> FN
        for rem in ans_map.values():
            for arow in rem:
                lf_counters['description']['fn'] += 1
                if pd.notna(arow['quantity']):
                    lf_counters['quantity']['fn'] += 1
                if pd.notna(arow['unit_price']):
                    lf_counters['unit_price']['fn'] += 1
                if pd.notna(arow['total']):
                    lf_counters['total']['fn'] += 1

    # Compute line-item metrics
    li_metrics = {}
    for f, c in lf_counters.items():
        li_metrics[f] = compute_metrics(c['tp'], c['fp'], c['fn'])

    st.subheader("Line-item metrics")
    st.json(li_metrics)

    avg_prec_li = sum(m['precision'] for m in li_metrics.values()) / len(li_metrics)
    avg_rec_li = sum(m['recall'] for m in li_metrics.values()) / len(li_metrics)
    avg_f1_li = sum(m['f1'] for m in li_metrics.values()) / len(li_metrics)
    st.write(f"Line-item average Precision: {avg_prec_li:.3f}, Recall: {avg_rec_li:.3f}, F1: {avg_f1_li:.3f}")

    st.success("Evaluation complete.")
