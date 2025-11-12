"""
03 - regex lineitems
"""

import os
import re
import pandas as pd
from src.regex_invoices import process_raw_data  

def ExtractRawLineItems(file_path: str):
    """
    Extract invoice-level and line-item-level data from a raw file.

    Args:
        file_path (str): Path to the raw input file.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (invoices_df, lineitems_df)
    """

    # Step 1: Get cleaned invoice data from ProcessRawData()
    df = process_raw_data(file_path)

    # Remove unwanted columns if present
    df = df.drop(columns=["id", "line_item_count"], errors="ignore")
    


    invoices_data = []
    lineitems_data = []

    # Helper function to clean line item descriptions
    def clean_description(desc: str) -> str:
        desc = desc.lower().strip()
        desc = desc.replace("\n", " ")  # remove newline characters
        desc = re.sub(r"\b\d+\s*[x#]\s*\d*\s*(oz|lb|cs|ea|bn|bag)?\b", "", desc)
        desc = re.sub(r"\b(cs|ea|bn|bag|lb|oz|#)\b", "", desc)
        desc = re.sub(r"[^a-z0-9\s,]+", " ", desc)
        desc = re.sub(r"\s+", " ", desc)
        return desc.strip()


    cleaned_text_temp = clean_description(df.loc[0,'cleaned_text'])
    
    # Regex pattern for extracting line items
    line_pattern = re.compile(
        r"(\d{4,6})?\s*([\d\.]+)\s+(?:ea|cs|bn|bag|lb|oz|#)?\s*([A-Za-z0-9\s\+\-,#&/]+?)\s+(\d+\.\d{2})",
        re.IGNORECASE
    )

    # Step 2: Process each record in the dataframe
    for _, row in df.iterrows():
        text = str(row.get("cleaned_text", "") or row.get("clean_text", ""))
        file_path_value = row.get("file_path", "")
        vendor = row.get("vendor", "")
        date = row.get("date", "")
        total = row.get("total", "")
        invoice_number = row.get("invoice_number", "")

        # Invoice-level record
        invoices_data.append({
            "file_path": file_path_value,
            "vendor": vendor,
            "date": date,
            "total": total,
            "invoice_number": invoice_number,
            "cleaned_text": cleaned_text_temp
        })

        # Step 3: Extract line items from invoice text
        matches = line_pattern.findall(text)
        if matches:
            for match in matches:
                _, qty, desc, total_price = match
                try:
                    qty = int(float(qty))
                except:
                    qty = 0
                total_price = total_price.strip()
                try:
                    unit_price = round(float(total_price) / float(qty), 2) if qty != 0 else ""
                except:
                    unit_price = ""
                clean_desc = clean_description(desc)
                lineitems_data.append({
                    "file_path": file_path_value,
                    "invoice_number": invoice_number,
                    "description": clean_desc,
                    "quantity": qty,
                    "unit_price": unit_price,
                    "total": total_price
                })
        else:
            # Add empty entry if no matches found
            lineitems_data.append({
                "file_path": file_path_value,
                "invoice_number": invoice_number,
                "description": "",
                "quantity": 0,
                "unit_price": "",
                "total": ""
            })

    # Step 4: Convert to DataFrames
    invoices_df = pd.DataFrame(invoices_data)
    lineitems_df = pd.DataFrame(lineitems_data)

    return invoices_df, lineitems_df

# Example Usage
file_path = r".\data\raw\train\Copy of ARPFIINVOEBTCHLASER (1).pdf"
invoices_df, lineitems_df = ExtractRawLineItems(file_path)

print("Invoices:")
print(invoices_df.head())

print("\nLine Items:")
print(lineitems_df.head())