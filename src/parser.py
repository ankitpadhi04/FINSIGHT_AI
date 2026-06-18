# src/parser.py

import pdfplumber
import pandas as pd
import re
from datetime import datetime


def parse_pdf(file) -> pd.DataFrame:
    with pdfplumber.open(file) as pdf:
        full_text  = "\n".join(page.extract_text() or "" for page in pdf.pages)
        all_tables = []
        for page in pdf.pages:
            t = page.extract_tables()
            if t:
                all_tables.extend(t)

    app_type = detect_app(full_text)

    if app_type == "gpay":
        return parse_gpay(full_text)
    elif app_type == "paytm":
        return parse_paytm(full_text, all_tables)
    else:
        return parse_gpay(full_text)


def detect_app(text: str) -> str:
    text_lower = text.lower()
    if "paytm" in text_lower:
        return "paytm"
    if "google" in text_lower or "gpay" in text_lower or "paitdo" in text_lower:
        return "gpay"
    return "gpay"


# ── GPay parser ───────────────────────────────────────────────

def parse_gpay(text: str) -> pd.DataFrame:
    pattern = re.compile(
        r'(\d{1,2}May2[,.]?\d{3,4})'
        r'\s+'
        r'(Paitdo|Receivfedrom|ReceifvreodR?|Receifvreod)'
        r'([A-Za-z0-9\s]+?)'
        r'\s*₹([\d,]+(?:\.\d{1,2})?)'
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

        date_clean = re.sub(
            r'(\d+)(May)2[,.]?(\d+)',
            lambda m: f"{m.group(1)} {m.group(2)} 2{m.group(3)}",
            date_raw
        )
        date_clean = date_clean.replace(",", "").strip()

        is_credit        = txn_type.startswith("Receif") or txn_type.startswith("Receivf")
        transaction_type = "credit" if is_credit else "debit"

        name_clean = re.sub(r'\d+', '', name).strip()
        if not name_clean:
            name_clean = name.strip()

        transactions.append({
            "date":             date_clean,
            "description":      name_clean,
            "amount":           amount,
            "transaction_type": transaction_type
        })

    return build_dataframe(transactions)


# ── Paytm parser ──────────────────────────────────────────────

def extract_paytm_year(text: str) -> str:
    short_year = re.search(r"'(\d{2})", text)
    if short_year:
        return "20" + short_year.group(1)
    full_year = re.search(r'20(2\d)', text)
    if full_year:
        return "20" + full_year.group(1)
    return str(datetime.now().year)


def parse_paytm_date(date_str: str, year: str) -> str:
    date_str = date_str.strip()
    months   = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
    for m in months:
        if m.lower() in date_str.lower():
            if year not in date_str:
                date_str = date_str + " " + year
            break
    return date_str


def extract_paytm_name(detail_line: str) -> str:
    detail_line = detail_line.strip()
    lower = detail_line.lower()
    if lower.startswith("paid to "):
        return detail_line[8:].strip()
    if lower.startswith("received from "):
        return detail_line[14:].strip()
    if lower.startswith("paid to"):
        return detail_line[7:].strip()
    if lower.startswith("received from"):
        return detail_line[13:].strip()
    return detail_line


def parse_paytm(text: str, tables: list) -> pd.DataFrame:
    year = extract_paytm_year(text)

    # Primary: parse from raw text line by line
    # Each transaction line looks like:
    # "17 Jun Paid to SWIGGY Tag: State Bank - Rs.1"
    # "17 Jun Received from Akash Pani Note: UPI State Bank + Rs.40"
    transactions = parse_paytm_text(text, year)

    # Fallback: table-based extraction for anything missed
    table_transactions = parse_paytm_tables(tables, year)

    # Merge and deduplicate
    all_txns = transactions + table_transactions
    if not all_txns:
        return build_dataframe([])

    df_combined = pd.DataFrame(all_txns)
    df_combined = df_combined.drop_duplicates(
        subset=["date", "description", "amount", "transaction_type"]
    )
    return build_dataframe(df_combined.to_dict(orient="records"))


def parse_paytm_text(text: str, year: str) -> list:
    """
    Primary Paytm parser using raw text.

    Each transaction appears on one line like:
        "17 Jun Paid to SWIGGY Tag: State Bank - Rs.1"
        "17 Jun Received from Akash Pani Note: UPI State Bank + Rs.40"
        "17 Jun Received from Santosh Kumar Padhi Note: UPI State Bank + Rs.1,200"

    Pattern: DD Mon  (Paid to | Received from) NAME  ... (+|-) Rs.AMOUNT
    """
    transactions = []

    # Match the full transaction line
    pattern = re.compile(
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))'  # date
        r'\s+'
        r'((?:Paid to|Received from)\s+(.+?))'                               # action + name
        r'\s+(?:Note:|Tag:|\w+:).+'                                           # notes section
        r'\s+([+\-])\s*Rs\.([\d,]+)',                                         # amount
        re.IGNORECASE
    )

    for match in pattern.finditer(text):
        date_raw   = match.group(1).strip()
        action     = match.group(2).strip()
        sign       = match.group(4)
        amount_str = match.group(5).replace(",", "")

        try:
            amount = float(amount_str)
        except ValueError:
            continue

        if amount <= 0:
            continue

        date_clean  = parse_paytm_date(date_raw, year)
        description = extract_paytm_name(action)
        is_credit   = sign == "+"

        transactions.append({
            "date":             date_clean,
            "description":      description,
            "amount":           amount,
            "transaction_type": "credit" if is_credit else "debit"
        })

    return transactions


def parse_paytm_tables(tables: list, year: str) -> list:
    """Fallback table-based extraction."""
    transactions = []

    for table in tables:
        for row in table:
            if not row or len(row) < 5:
                continue

            date_cell   = str(row[0] or "").strip()
            detail_cell = str(row[1] or "").strip()
            amount_cell = str(row[4] or "").strip()

            if not amount_cell or "Amount" in amount_cell:
                continue
            if not detail_cell or "Transaction" in detail_cell:
                continue
            if "Passbook" in detail_cell:
                continue
            if "Rs." not in amount_cell:
                continue

            is_credit = amount_cell.strip().startswith("+")
            amt_clean = re.sub(r'[+\-Rs.,\s]', '', amount_cell)
            try:
                amount = float(amt_clean)
            except ValueError:
                continue

            if amount <= 0:
                continue

            date_line   = date_cell.split("\n")[0].strip()
            date_clean  = parse_paytm_date(date_line, year)
            detail_line = detail_cell.split("\n")[0].strip()
            description = extract_paytm_name(detail_line)

            if not description:
                continue

            transactions.append({
                "date":             date_clean,
                "description":      description,
                "amount":           amount,
                "transaction_type": "credit" if is_credit else "debit"
            })

    return transactions


# ── Shared utilities ──────────────────────────────────────────

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
## End of src/parser.py
