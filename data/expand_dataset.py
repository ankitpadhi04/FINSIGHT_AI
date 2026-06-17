# data/expand_dataset.py

import pandas as pd
import random

random.seed(42)

base_data = pd.read_csv("data/transactions.csv")

# ── Semantic context templates per category ──────────────────
# These teach DistilBERT meaning, not just spelling
CONTEXT_TEMPLATES = {
    "Food": [
        "food stall payment", "restaurant bill", "meal delivery",
        "breakfast payment", "lunch order", "dinner payment",
        "chai shop", "tiffin payment", "canteen bill", "dhaba payment",
        "bakery shop", "sweet shop", "snacks stall"
    ],
    "Travel": [
        "cab fare", "auto payment", "bus ticket", "train booking",
        "petrol fill", "fuel payment", "parking charges",
        "toll booth", "bike taxi", "flight booking", "metro recharge"
    ],
    "Shopping": [
        "clothes shop", "online order", "delivery payment",
        "electronics purchase", "mobile shop", "garment store",
        "footwear payment", "accessories purchase", "tailor fees"
    ],
    "Entertainment": [
        "movie ticket", "streaming subscription", "gaming payment",
        "concert ticket", "ott subscription", "music app",
        "show booking", "amusement park", "entertainment app"
    ],
    "Health": [
        "medicine purchase", "doctor fees", "hospital bill",
        "pharmacy payment", "lab test fees", "clinic payment",
        "health checkup", "gym fees", "yoga class", "medical store"
    ],
    "Bills": [
        "electricity bill", "mobile recharge", "wifi payment",
        "gas bill", "water bill", "dth recharge", "postpaid bill",
        "insurance emi", "loan payment", "credit card bill"
    ],
    "Groceries": [
        "kirana shop", "vegetable purchase", "grocery store",
        "milk payment", "daily needs", "provisions store",
        "supermarket bill", "rice purchase", "sabji market"
    ],
    "Education": [
        "college fees", "tuition payment", "online course",
        "exam fees", "coaching center", "book purchase",
        "school fees", "library fees", "study material"
    ],
    "Transfer": [
        "sent to friend", "personal transfer", "rent payment",
        "money returned", "family transfer", "split bill",
        "pg rent", "hostel fees", "group payment"
    ],
    "Investment": [
        "mutual fund", "sip payment", "stock purchase",
        "fd deposit", "gold investment", "insurance premium",
        "ppf deposit", "nps contribution", "rd deposit"
    ]
}


def garble(name: str) -> str:
    """Simulate GPay-style garbled text."""
    name = list(name)
    for i in range(len(name)):
        if random.random() < 0.2 and name[i].isalpha():
            name[i] = name[i].upper() if name[i].islower() else name[i].lower()
    return "".join(name)


def add_suffix(name: str) -> str:
    suffixes = ["Payment", "Pay", "Pvt", "Ltd", "Store",
                "Shop", "App", "Online", "Center", "Mart"]
    return name + random.choice(suffixes)


augmented = []

# 1. Original + garbled + suffix variants
for _, row in base_data.iterrows():
    desc = row["description"]
    cat  = row["category"]
    augmented.append({"description": desc, "category": cat})
    for _ in range(4):
        augmented.append({"description": garble(desc), "category": cat})
    augmented.append({"description": add_suffix(desc), "category": cat})
    augmented.append({"description": garble(add_suffix(desc)), "category": cat})

# 2. Context template examples
for cat, templates in CONTEXT_TEMPLATES.items():
    for t in templates:
        augmented.append({"description": t, "category": cat})
        augmented.append({"description": garble(t), "category": cat})

df = pd.DataFrame(augmented)
df = df.drop_duplicates(subset=["description"])
df = df[df["description"].str.strip() != ""]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("data/transactions_augmented.csv", index=False)

print(f"Total samples: {len(df)}")
print("\nPer category:")
print(df["category"].value_counts())