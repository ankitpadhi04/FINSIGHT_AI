# src/parser.py

import pdfplumber
import pandas as pd
import re


def parse_pdf(file) -> pd.DataFrame:
    """
    Main entry point. Extracts transactions from GPay/PhonePe PDFs.
    Returns DataFrame with columns: date, description, amount, transaction_type
    """
    with pdfplumber.open(file) as pdf:
        full_text = "\n".join(
            page.extract_text() or "" for page in pdf.pages
        )

    app_type = detect_app(full_text)

    if app_type == "gpay":
        return parse_gpay(full_text)
    elif app_type == "phonepe":
        return parse_phonepe(full_text)
    else:
        return parse_gpay(full_text)  # default to gpay pattern


def detect_app(text: str) -> str:
    text_lower = text.lower()
    if "google" in text_lower or "gpay" in text_lower or "paitdo" in text_lower:
        return "gpay"
    elif "phonepe" in text_lower or "phone pe" in text_lower:
        return "phonepe"
    return "gpay"


def parse_gpay(text: str) -> pd.DataFrame:
    """
    Parses garbled GPay PDF text using regex on the known broken pattern.

    Each transaction line looks like one of:
        04May2,026 PaitdoPINTDUUTTA ₹6
        27May2,026 ReceifvreodAmbhinWaavlde ₹30

    The garbling is consistent:
        'Paid to'     becomes  'Paitdo'
        'Received from' becomes 'ReceifvreodNAME' or 'Receivfedrom'
    """

    # Match: date | Paitdo/Receivfed + name | ₹amount
    pattern = re.compile(
        r'(\d{1,2}May2[,.]?\d{3,4})'          # date e.g. 04May2,026
        r'\s+'
        r'(Paitdo|Receivfedrom|ReceifvreodR?|Receifvreod)'  # transaction keyword
        r'([A-Za-z0-9\s]+?)'                   # merchant/person name
        r'\s*₹([\d,]+(?:\.\d{1,2})?)'         # amount
    )

    transactions = []
    for match in pattern.finditer(text):
        date_raw   = match.group(1)
        txn_type   = match.group(2)
        name       = match.group(3).strip()
        amount_str = match.group(4).replace(",", "")

        try:
            amount = float(amount_str)
        except ValueError:
            continue

        if amount <= 0:
            continue

        # Normalize date: "04May2,026" → "04 May 2026"
        date_clean = re.sub(r'(\d+)(May)2[,.]?(\d+)', lambda m: f"{m.group(1)} {m.group(2)} 2{m.group(3)}", date_raw)
        date_clean = date_clean.replace(",", "").strip()

        # Normalize transaction type
        is_credit = txn_type.startswith("Receif") or txn_type.startswith("Receivf")
        transaction_type = "credit" if is_credit else "debit"

        # Clean up name (remove leftover digit noise)
        name_clean = re.sub(r'\d+', '', name).strip()
        if not name_clean:
            name_clean = name.strip()

        transactions.append({
            "date": date_clean,
            "description": name_clean,
            "amount": amount,
            "transaction_type": transaction_type
        })

    return build_dataframe(transactions)


def parse_phonepe(text: str) -> pd.DataFrame:
    """
    PhonePe parser — placeholder, extend when you have a PhonePe PDF.
    Falls back to gpay pattern for now.
    """
    return parse_gpay(text)


def build_dataframe(transactions: list) -> pd.DataFrame:
    if not transactions:
        return pd.DataFrame(
            columns=["date", "description", "amount", "transaction_type"]
        )

    df = pd.DataFrame(transactions)
    df = df[df["amount"] > 0].copy()
    df = df.drop_duplicates(subset=["date", "description", "amount"])
    df = df.sort_values("date").reset_index(drop=True)
    return df