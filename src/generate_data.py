"""
WIE3007 Data Mining - AML Suspicious Transaction Detection
Synthetic dataset generator (Member A).

Produces ~5,000 bank transactions across 800 accounts, Jan-Jun 2026, with ~4% suspicious
transactions created from five planted money-laundering typologies, plus "hard negative"
legitimate behaviour (large but genuine business, tuition and family-remittance flows)
so the task is not trivially easy.

Text columns (transaction_memo, kyc_note) were authored with GenAI (Claude) as
template pools, then sampled and combined here with deliberate noise:
  * ~30% of suspicious transactions carry ordinary-looking memos
  * ~8% of normal transactions carry vague memos
  * ~40% of suspicious accounts have a benign KYC note; ~6% of normal accounts a concerning one

Usage:  python generate_data.py  -> data/raw/raw_transactions.csv + data/raw/typology_key.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)
N_TXN = 5000
START, END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-30 23:59")
CASH_THRESHOLD = 25_000  # Malaysia cash threshold reporting level (RM)

# --------------------------------------------------------------------------------------
# Occupations: (name, income range RM/month, cash_intensive)
# --------------------------------------------------------------------------------------
RETAIL_OCC = [("Engineer", 5000, 12000), ("Teacher", 3500, 7000), ("Nurse", 3200, 6500),
              ("Clerk", 2200, 4000), ("Accountant", 4500, 10000), ("Software Developer", 5500, 14000),
              ("Sales Executive", 3000, 6500), ("Civil Servant", 3000, 7500), ("Technician", 2500, 4800),
              ("Lecturer", 6000, 12000), ("Pharmacist", 6000, 11000), ("Police Officer", 3200, 6000),
              ("Graphic Designer", 3000, 6500), ("Bank Officer", 4000, 8500), ("Doctor", 9000, 25000),
              ("Driver (e-hailing)", 2000, 4500), ("Retiree", 1500, 4000), ("Homemaker", 0, 1500)]
SME_OCC = [("Restaurant Owner", 8000, 30000), ("Car Dealer", 12000, 45000), ("Contractor", 10000, 40000),
           ("Trader (Import/Export)", 12000, 50000), ("Retail Shop Owner", 6000, 20000),
           ("Money Changer", 10000, 35000), ("Jewellery Shop Owner", 12000, 40000),
           ("Hardware Shop Owner", 7000, 22000), ("Online Seller", 4000, 15000)]
STUDENT_OCC = [("Student", 0, 1200)]
HNW_OCC = [("Company Director", 40000, 150000), ("Property Investor", 30000, 120000),
           ("Medical Specialist", 35000, 90000)]
LOW_OCC = [("Student", 0, 1200), ("Unemployed", 0, 800), ("Part-time Worker", 800, 2000),
           ("Factory Operator", 1700, 2600), ("Clerk", 2200, 4000), ("Homemaker", 0, 1500)]

# --------------------------------------------------------------------------------------
# Memo pools (GenAI-authored)
# --------------------------------------------------------------------------------------
MEMO = {
    "salary_in": ["SALARY {m}", "Gaji {m}", "PAYROLL {M}", "Salary credit {m}", "Monthly salary"],
    "family_in": ["duit dari mak", "from abang", "family support", "pocket money", "ayah bank in",
                  "duit belanja", "allowance {m}", "angpau", "raya money", "from sis"],
    "refund_in": ["refund deposit", "split bill dinner", "share grab", "return money", "bayar balik hutang",
                  "your share of trip", "insurance claim", "refund shopee"],
    "business_in": ["INV-{inv} payment", "Payment for INV {inv}", "Customer payment PO{inv}", "Sales {m}",
                    "Supply payment {m}", "Settlement INV{inv}", "Deposit for order {inv}",
                    "Car sale balance {inv}", "Progress claim {inv}"],
    "personal_out": ["rent {m}", "sewa rumah", "bayar hutang", "car loan", "makan", "share grab", "insurance",
                     "phone bill", "gym fee", "kids tuition", "nursery fee", "duit belanja mak", "for parents",
                     "wedding gift", "wifi bill", "PTPTN", "tuition adik", "groceries", "birthday gift",
                     "lunch money", "parking", "house renovation", "netflix share", "concert ticket"],
    "business_out": ["Supplier INV {inv}", "Stock purchase {m}", "Payment to vendor {inv}", "Staff salary {m}",
                     "Rental shop lot {m}", "Raw material {inv}", "Logistics charge {inv}", "SST payment {m}"],
    "wire_tuition": ["Tuition fee semester", "University fees", "Student accommodation", "Course fee"],
    "wire_family": ["family support", "kiriman keluarga", "for mother", "monthly support", "house repair kampung"],
    "wire_business": ["Import payment PO{inv}", "Machinery deposit", "Supplier payment INV{inv}",
                      "Trade settlement {inv}"],
    "cash_dep": ["cash deposit", "kedai collection", "daily sales", "deposit", "", "takings {m}"],
    "cash_wd": ["", "cash", "withdraw", "pocket money"],
    "ewallet": ["TNG reload", "e-wallet topup", "Grab wallet", "ShopeePay topup", "Boost reload"],
    "bill": ["TNB bill", "Air Selangor", "Unifi bill", "Astro", "Celcom bill", "Maxis bill", "cukai pintu"],
    # vague / red-flag style memos
    "vague": ["payment", "transfer", "for project", "consultancy fee", "investment return", "help friend",
              "urgent", "loan", "ok", "-", "goods", "commission", "as discussed", "pls check", "service fee",
              "deal", "profit share", "trading", "package", "for boss", "settle", "misc", "advance",
              "crypto topup", "USDT", "agent fee", "release fund", "capital injection", "return capital",
              "", "tq", "done"],
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]


def fill(t, ts):
    m = MONTHS[ts.month - 1]
    return (t.replace("{m}", m).replace("{M}", m.upper())
             .replace("{inv}", str(rng.integers(1000, 99999))))


def memo(kind, ts):
    return fill(rng.choice(MEMO[kind]), ts)


# --------------------------------------------------------------------------------------
# KYC note fragments (GenAI-authored)
# --------------------------------------------------------------------------------------
KYC_BENIGN = [
    "Long-standing customer, salary credited monthly, profile consistent with declared occupation.",
    "Customer verified in branch with MyKad and payslip. No adverse findings.",
    "Account used mainly for salary, bills and family transfers. Low risk.",
    "Documents complete. Employment letter on file. Activity in line with income.",
    "Customer is a regular walk-in, friendly and cooperative during periodic review.",
    "Business registered with SSM; turnover broadly consistent with bank statements provided.",
    "Student account opened with university offer letter; parents provide monthly support.",
    "Retired customer, pension and dividends credited. Simple usage pattern.",
    "Periodic review completed. No change in risk profile.",
    "SME customer with steady supplier and customer payments; financials submitted on time.",
    "Customer updated address and phone number; supporting documents verified.",
    "Joint relationship with spouse; usage mostly household expenses.",
    "Online-only customer, eKYC passed, low and regular activity.",
    "High-net-worth client managed by relationship manager; wealth source documented (business sale).",
]
KYC_CONCERN = [
    "Customer was reluctant to explain source of funds for recent large deposits.",
    "Frequent cash deposits not consistent with stated occupation; follow-up recommended.",
    "Account opened recently; several third parties seen depositing into this account.",
    "Customer gave inconsistent answers about employer during verification call.",
    "Address verification still pending; mail returned undelivered.",
    "Customer requested to split deposit into smaller amounts at the counter.",
    "Funds received are moved out quickly; customer said it is 'for a friend's business'.",
    "Declared income is low but account turnover is high; enhanced due diligence suggested.",
    "Customer could not provide invoices for overseas payments when asked.",
    "Customer appeared to be accompanied and instructed by another person at the branch.",
    "Multiple ATM cards requested within a short period; reason unclear.",
    "Overseas transfers to unfamiliar counterparties; customer described them as 'investment'.",
]
KYC_MILD = [  # used for hard-negative noise on normal accounts
    "Customer occasionally deposits large cash amounts from business takings; explained and documented.",
    "Some overseas transfers for children's education; supporting letters provided.",
    "Recent increase in activity explained by side business; monitoring as routine.",
]

# --------------------------------------------------------------------------------------
# Accounts
# --------------------------------------------------------------------------------------
SUS_COUNTS = {"structuring": 12, "pass_through": 10, "mule": 14, "high_risk_wire": 10, "income_mismatch": 10}
N_ACC = 800


def pick_occ(pool):
    o = pool[rng.integers(len(pool))]
    inc = 0 if o[2] == 0 else int(round(rng.uniform(o[1], o[2]), -1))
    return o[0], inc


accounts = []
aid = 1


def new_acc(segment, typology, occ_pool, open_from, open_to):
    global aid
    occ, inc = pick_occ(occ_pool)
    span = (open_to - open_from).days
    open_date = open_from + pd.Timedelta(days=int(rng.integers(0, max(span, 1))))
    a = dict(account_id=f"ACC{aid:05d}", segment=segment, typology=typology, occupation=occ,
             declared_monthly_income=inc, open_date=open_date)
    aid += 1
    accounts.append(a)
    return a


n_sus = sum(SUS_COUNTS.values())
plan = (["retail"] * 520 + ["sme"] * 120 + ["student"] * 70 + ["hnw"] * 30)[: N_ACC - n_sus]
for seg in plan:
    pool = {"retail": RETAIL_OCC, "sme": SME_OCC, "student": STUDENT_OCC, "hnw": HNW_OCC}[seg]
    lo = pd.Timestamp("2023-06-01") if seg == "student" else pd.Timestamp("2012-01-01")
    new_acc(seg, "normal", pool, lo, pd.Timestamp("2025-12-01"))
for typ, k in SUS_COUNTS.items():
    for _ in range(k):
        if typ == "mule":
            new_acc("retail", typ, LOW_OCC[:4], pd.Timestamp("2025-12-15"), pd.Timestamp("2026-04-15"))
        elif typ == "income_mismatch":
            new_acc("retail", typ, LOW_OCC, pd.Timestamp("2018-01-01"), pd.Timestamp("2025-09-01"))
        elif typ == "structuring":
            pool = RETAIL_OCC if rng.random() < 0.6 else SME_OCC
            new_acc("retail" if pool is RETAIL_OCC else "sme", typ, pool,
                    pd.Timestamp("2015-01-01"), pd.Timestamp("2025-10-01"))
        else:
            pool = SME_OCC if rng.random() < 0.5 else RETAIL_OCC
            new_acc("sme" if pool is SME_OCC else "retail", typ, pool,
                    pd.Timestamp("2014-01-01"), pd.Timestamp("2025-10-01"))

acc = {a["account_id"]: a for a in accounts}

# --------------------------------------------------------------------------------------
# Transactions
# --------------------------------------------------------------------------------------
rows = []
cp_id = [1]


def counterparty(prefix="CP"):
    cp_id[0] += 1
    return f"{prefix}{cp_id[0]:06d}"


def rand_ts(lo=START, hi=END, night=0.08):
    lo = max(lo, START)
    if lo >= hi:
        lo = hi - pd.Timedelta(days=1)
    d = lo + pd.Timedelta(seconds=int(rng.integers(0, int((hi - lo).total_seconds()))))
    if rng.random() < night:
        hour = int(rng.choice([0, 1, 2, 3, 4, 23]))
    else:
        hour = int(np.clip(rng.normal(14, 3.5), 7, 22))
    return d.replace(hour=hour, minute=int(rng.integers(60)), second=int(rng.integers(60)))


def add(a, ts, amount, ttype, channel, xb=0, risk="none", memo_txt="", cp=None, label=0, typ="normal"):
    rows.append(dict(account_id=a["account_id"], timestamp=ts, amount_myr=round(float(amount), 2),
                     txn_type=ttype, channel=channel, is_cross_border=xb, dest_country_risk=risk,
                     counterparty_id=cp or counterparty(), transaction_memo=memo_txt,
                     is_suspicious=label, typology=typ))


def logn(median, sigma):
    return median * np.exp(rng.normal(0, sigma))


DIGITAL = ["online_banking", "mobile_app"]


def normal_txn(a, hi=END):
    """One ordinary transaction appropriate to the account's segment."""
    ts = rand_ts(max(a["open_date"], START), hi)
    seg, inc = a["segment"], max(a["declared_monthly_income"], 800)
    r = rng.random()
    if seg == "sme":
        if r < 0.30 and a["occupation"] in ("Trader (Import/Export)", "Online Seller", "Car Dealer") and rng.random() < 0.35:
            add(a, ts, logn(inc * 0.9, 0.6), "wire_in", "online_banking", 1,
                rng.choice(["low", "medium"], p=[0.6, 0.4]),
                rng.choice(["Export proceeds INV{inv}", "Buyer payment PO{inv}", "Overseas customer {inv}"]).replace(
                    "{inv}", str(rng.integers(1000, 99999))))
        elif r < 0.30:
            add(a, ts, logn(inc * 0.4, 0.7), "transfer_in", rng.choice(DIGITAL), memo_txt=memo("business_in", ts))
        elif r < 0.55:
            add(a, ts, logn(inc * 0.3, 0.7), "transfer_out", rng.choice(DIGITAL), memo_txt=memo("business_out", ts))
        elif r < 0.72:
            amt = logn(inc * 0.35, 0.6)
            if a["occupation"] in ("Car Dealer", "Jewellery Shop Owner", "Money Changer") and rng.random() < 0.35:
                amt = rng.uniform(26_000, 90_000)  # hard negative: large, above threshold, legitimate
            add(a, ts, amt, "cash_deposit", rng.choice(["branch", "CDM"], p=[0.6, 0.4]), memo_txt=memo("cash_dep", ts))
        elif r < 0.80 and a["occupation"] == "Trader (Import/Export)" or r < 0.76:
            risk = rng.choice(["low", "medium"], p=[0.6, 0.4])
            add(a, ts, logn(inc * 0.8, 0.6), "wire_out", "online_banking", 1, risk, memo("wire_business", ts))
        elif r < 0.90:
            add(a, ts, logn(1500, 0.6), "bill_payment", rng.choice(DIGITAL), memo_txt=memo("bill", ts))
        else:
            add(a, ts, logn(2500, 0.8), "cash_withdrawal", rng.choice(["ATM", "branch"]), memo_txt=memo("cash_wd", ts))
        return
    if seg == "hnw":
        if r < 0.08:
            add(a, ts, logn(inc * 0.8, 0.6), "wire_in", "online_banking", 1, rng.choice(["low", "medium"], p=[0.85, 0.15]),
                rng.choice(["Overseas dividend", "Foreign rental income", "Investment redemption", "Consulting fee SG"]))
        elif r < 0.25:
            add(a, ts, logn(inc * 0.6, 0.6), "transfer_in", rng.choice(DIGITAL),
                memo_txt=rng.choice(["Dividend", "Rental income", "Director fee", "Unit trust redemption", "Bonus"]))
        elif r < 0.55:
            add(a, ts, logn(inc * 0.3, 0.8), "transfer_out", rng.choice(DIGITAL),
                memo_txt=rng.choice(["Property instalment", "Unit trust", "Kids school fee", "Car instalment",
                                     "Renovation", "Insurance premium", "Charity donation"]))
        elif r < 0.75:
            risk = rng.choice(["low", "medium"], p=[0.8, 0.2])
            add(a, ts, logn(inc * 0.7, 0.7), "wire_out", "online_banking", 1, risk,
                rng.choice(MEMO["wire_tuition"] + ["Overseas property", "Investment account top-up", "Holiday villa"]))
        elif r < 0.88:
            add(a, ts, logn(1200, 0.6), "bill_payment", "online_banking", memo_txt=memo("bill", ts))
        else:
            add(a, ts, logn(4000, 0.7), "cash_withdrawal", "branch", memo_txt=memo("cash_wd", ts))
        return
    if seg == "student":
        if r < 0.35:
            add(a, ts, logn(500, 0.5), "transfer_in", "mobile_app", memo_txt=memo("family_in", ts))
        elif r < 0.65:
            add(a, ts, logn(60, 0.8), "transfer_out", "mobile_app", memo_txt=memo("personal_out", ts))
        elif r < 0.85:
            add(a, ts, logn(50, 0.5), "e_wallet_topup", "mobile_app", memo_txt=memo("ewallet", ts))
        else:
            add(a, ts, logn(100, 0.5), "cash_withdrawal", "ATM", memo_txt=memo("cash_wd", ts))
        return
    # retail
    if r < 0.17:
        add(a, ts.replace(day=min(ts.day, 28)), inc * rng.uniform(0.95, 1.0), "transfer_in", "online_banking",
            memo_txt=memo("salary_in", ts), cp=f"EMP{hash(a['account_id']) % 9000 + 1000}")
    elif r < 0.25:
        add(a, ts, logn(200, 0.9), "transfer_in", rng.choice(DIGITAL), memo_txt=memo("refund_in", ts))
    elif r < 0.55:
        add(a, ts, logn(min(inc * 0.08, 1500), 0.9), "transfer_out", rng.choice(DIGITAL), memo_txt=memo("personal_out", ts))
    elif r < 0.66:
        add(a, ts, logn(80, 0.6), "e_wallet_topup", "mobile_app", memo_txt=memo("ewallet", ts))
    elif r < 0.80:
        add(a, ts, logn(220, 0.5), "bill_payment", rng.choice(DIGITAL), memo_txt=memo("bill", ts))
    elif r < 0.92:
        add(a, ts, logn(300, 0.6), "cash_withdrawal", "ATM", memo_txt=memo("cash_wd", ts))
    elif r < 0.97:
        add(a, ts, logn(800, 0.8), "cash_deposit", rng.choice(["CDM", "branch"]), memo_txt=memo("cash_dep", ts))
    elif r < 0.985:  # legitimate inbound wire (overseas salary / family abroad)
        add(a, ts, logn(4000, 0.6), "wire_in", "online_banking", 1, rng.choice(["low", "medium"], p=[0.7, 0.3]),
            rng.choice(["Salary SG", "Freelance payment", "from son in Australia", "family support", "Upwork payment",
                        "Refund overseas", ""]))
    else:  # hard negative: legitimate cross-border
        kind = rng.choice(["wire_tuition", "wire_family"])
        risk = "low" if kind == "wire_tuition" else rng.choice(["low", "medium"], p=[0.3, 0.7])
        amt = logn(15_000, 0.4) if kind == "wire_tuition" else logn(900, 0.5)
        add(a, ts, amt, "wire_out", "online_banking", 1, risk, memo(kind, ts))


def sus_memo(normal_kind, ts):
    """Suspicious txn memo: vague 70%, normal-looking 30% (noise)."""
    return memo("vague", ts) if rng.random() < 0.70 else memo(normal_kind, ts)


# ---- planted typologies ----
def structuring(a):
    base = rand_ts(START, END - pd.Timedelta(days=10))
    for _ in range(int(rng.integers(3, 6))):
        ts = rand_ts(base, base + pd.Timedelta(days=int(rng.integers(2, 8))), night=0.05)
        add(a, ts, rng.uniform(18_000, CASH_THRESHOLD - 50), "cash_deposit",
            rng.choice(["branch", "CDM"], p=[0.7, 0.3]), memo_txt=sus_memo("cash_dep", ts),
            label=1, typ="structuring")


def pass_through(a):
    ts = rand_ts(START, END - pd.Timedelta(days=3), night=0.35)
    amt = rng.uniform(60_000, 300_000)
    inbound = rng.choice(["wire_in", "transfer_in"])
    add(a, ts, amt, inbound, "online_banking", int(inbound == "wire_in"),
        rng.choice(["medium", "high"]) if inbound == "wire_in" else "none",
        sus_memo("business_in", ts), label=1, typ="pass_through")
    k = int(rng.integers(2, 5))
    parts = rng.dirichlet(np.ones(k)) * amt * rng.uniform(0.9, 0.99)
    for p_ in parts:
        t2 = ts + pd.Timedelta(minutes=int(rng.integers(20, 60 * 20)))
        add(a, t2, p_, rng.choice(["transfer_out", "wire_out"], p=[0.7, 0.3]), "online_banking",
            memo_txt=sus_memo("business_out", t2), label=1, typ="pass_through")
    # fix wire_out cross-border flags below


def mule(a):
    lo = max(a["open_date"] + pd.Timedelta(days=int(rng.integers(10, 60))), START)
    hi = min(lo + pd.Timedelta(days=5), END)
    total = 0
    for _ in range(int(rng.integers(3, 6))):
        ts = rand_ts(lo, hi, night=0.40)
        amt = rng.uniform(300, 4_500)
        total += amt
        add(a, ts, amt, "transfer_in", "mobile_app", memo_txt=sus_memo("refund_in", ts), label=1, typ="mule")
    ts = hi + pd.Timedelta(hours=int(rng.integers(1, 30)))
    ts = min(ts, END)
    out = rng.choice(["cash_withdrawal", "transfer_out", "e_wallet_topup"], p=[0.45, 0.4, 0.15])
    ch = "ATM" if out == "cash_withdrawal" else "mobile_app"
    add(a, ts, total * rng.uniform(0.85, 0.98), out, ch, memo_txt=sus_memo("personal_out", ts),
        label=1, typ="mule")


def high_risk_wire(a):
    for _ in range(int(rng.integers(1, 4))):
        ts = rand_ts(night=0.25)
        add(a, ts, logn(45_000, 0.7), "wire_out", "online_banking", 1,
            rng.choice(["high", "medium"], p=[0.8, 0.2]), sus_memo("wire_business", ts),
            label=1, typ="high_risk_wire")


def income_mismatch(a):
    for _ in range(int(rng.integers(2, 4))):
        ts = rand_ts(night=0.20)
        t = rng.choice(["transfer_in", "transfer_out", "cash_deposit"])
        ch = "branch" if t == "cash_deposit" else "online_banking"
        add(a, ts, rng.uniform(15_000, 120_000), t, ch, memo_txt=sus_memo(
            "refund_in" if t == "transfer_in" else "personal_out" if t == "transfer_out" else "cash_dep", ts),
            label=1, typ="income_mismatch")


TYPO_FN = dict(structuring=structuring, pass_through=pass_through, mule=mule,
               high_risk_wire=high_risk_wire, income_mismatch=income_mismatch)
for a in accounts:
    if a["typology"] != "normal":
        TYPO_FN[a["typology"]](a)
        n_bg = 1 if a["typology"] == "mule" else int(rng.integers(2, 5))
        for _ in range(n_bg):
            normal_txn(a)

# ---- legitimate look-alikes (hard negatives, label 0) ----
def lookalike_cash(a):  # cash-heavy business banking several near-threshold takings in a week
    base = rand_ts(START, END - pd.Timedelta(days=10))
    for _ in range(int(rng.integers(3, 6))):
        ts = rand_ts(base, base + pd.Timedelta(days=int(rng.integers(3, 8))), night=0.02)
        add(a, ts, rng.uniform(14_000, 24_900), "cash_deposit", rng.choice(["branch", "CDM"]),
            memo_txt=memo("cash_dep", ts), typ="lookalike_cash")


def lookalike_lumpsum(a):  # property sale / EPF / insurance maturity moved out quickly
    ts = rand_ts(START, END - pd.Timedelta(days=3))
    amt = rng.uniform(60_000, 280_000)
    add(a, ts, amt, "transfer_in", "online_banking",
        memo_txt=rng.choice(["Property sale proceeds", "EPF withdrawal", "Insurance maturity", "Inheritance",
                             "Land sale balance", "Fixed deposit uplift"]), typ="lookalike_lumpsum")
    k = int(rng.integers(2, 4))
    for p_ in rng.dirichlet(np.ones(k)) * amt * rng.uniform(0.6, 0.97):
        t2 = ts + pd.Timedelta(minutes=int(rng.integers(30, 60 * 22)))
        add(a, t2, p_, "transfer_out", "online_banking",
            memo_txt=rng.choice(["Developer payment", "Lawyer fee", "Car full settlement", "Housing loan settlement",
                                 "Unit trust", "Transfer to spouse", "Renovation contractor", "payment"]),
            typ="lookalike_lumpsum")


def lookalike_newseller(a):  # brand-new online seller: many small inbound payments, then pays supplier
    lo = max(a["open_date"] + pd.Timedelta(days=int(rng.integers(5, 40))), START)
    hi = min(lo + pd.Timedelta(days=6), END)
    tot = 0
    for _ in range(int(rng.integers(3, 7))):
        ts = rand_ts(lo, hi, night=0.15)
        v = rng.uniform(150, 3_500)
        tot += v
        add(a, ts, v, "transfer_in", "mobile_app",
            memo_txt=rng.choice(["order #{inv}".replace("{inv}", str(rng.integers(100, 999))), "payment", "COD",
                                 "baju raya", "tudung order", "preorder", "kek order", "deposit"]),
            typ="lookalike_newseller")
    ts = min(hi + pd.Timedelta(hours=int(rng.integers(2, 48))), END)
    add(a, ts, tot * rng.uniform(0.5, 0.9), rng.choice(["transfer_out", "cash_withdrawal"], p=[0.7, 0.3]),
        "mobile_app", memo_txt=rng.choice(["supplier", "stock", "payment", "kain supplier", "restock"]),
        typ="lookalike_newseller")


def lookalike_wire(a):  # genuine trade with higher-risk jurisdictions
    for _ in range(int(rng.integers(1, 3))):
        ts = rand_ts()
        add(a, ts, logn(35_000, 0.7), "wire_out", "online_banking", 1,
            rng.choice(["high", "medium"], p=[0.5, 0.5]), memo("wire_business", ts), typ="lookalike_wire")


normal_pool = [a for a in accounts if a["typology"] == "normal"]
order = rng.permutation(len(normal_pool))
idx = 0
for fn, seg, k in [(lookalike_cash, "sme", 14), (lookalike_lumpsum, "retail", 10),
                   (lookalike_wire, "sme", 10), (lookalike_lumpsum, "hnw", 4)]:
    chosen = [normal_pool[i] for i in order if normal_pool[i]["segment"] == seg][:k * 3]
    for a in rng.permutation(chosen)[:k]:
        if a.get("lookalike"):
            continue
        a["lookalike"] = True
        fn(a)
for _ in range(16):  # new online-seller accounts
    a = new_acc("retail", "normal", [("Online Seller", 2000, 9000), ("Student", 0, 1200), ("Part-time Worker", 800, 2000)],
                pd.Timestamp("2025-12-15"), pd.Timestamp("2026-04-15"))
    a["lookalike"] = True
    lookalike_newseller(a)
acc = {a["account_id"]: a for a in accounts}

# fill with normal activity up to N_TXN
normal_accs = [a for a in accounts if a["typology"] == "normal"]
weights = np.array([{"retail": 1.0, "sme": 1.6, "student": 0.7, "hnw": 1.2}[a["segment"]] for a in normal_accs])
weights /= weights.sum()
for a in normal_accs:  # every account at least one transaction
    normal_txn(a)
while len(rows) < N_TXN:
    normal_txn(normal_accs[rng.choice(len(normal_accs), p=weights)])
df = pd.DataFrame(rows)
if len(df) > N_TXN:  # trim surplus normal rows only
    drop = df[(df.is_suspicious == 0)].sample(len(df) - N_TXN, random_state=SEED).index
    df = df.drop(drop)

# wires: make sure cross-border flags/risks are consistent
w = df.txn_type.isin(["wire_out", "wire_in"])
df.loc[w, "is_cross_border"] = 1
df.loc[w & (df.dest_country_risk == "none"), "dest_country_risk"] = rng.choice(
    ["medium", "high"], size=int((w & (df.dest_country_risk == "none")).sum()))
df.loc[~w, ["is_cross_border", "dest_country_risk"]] = [0, "none"]

# --------------------------------------------------------------------------------------
# Account-level columns + behavioural features computed from history
# --------------------------------------------------------------------------------------
df = df.sort_values(["account_id", "timestamp"]).reset_index(drop=True)
df["occupation"] = df.account_id.map(lambda x: acc[x]["occupation"])
df["declared_monthly_income"] = df.account_id.map(lambda x: acc[x]["declared_monthly_income"])
df["account_age_days"] = (df.timestamp - df.account_id.map(lambda x: acc[x]["open_date"])).dt.days.clip(lower=0)

cnt24, rec7, avg30 = [], [], []
for _, g in df.groupby("account_id", sort=False):
    ts = g.timestamp.values
    amt = g.amount_myr.values
    cps = g.counterparty_id.values
    for i in range(len(g)):
        t = ts[i]
        m24 = (ts <= t) & (ts > t - np.timedelta64(24, "h"))
        m7 = (ts <= t) & (ts > t - np.timedelta64(7, "D"))
        m30 = (ts < t) & (ts > t - np.timedelta64(30, "D"))
        cnt24.append(int(m24.sum()))
        rec7.append(len(set(cps[m7])))
        avg30.append(round(float(amt[m30].mean()), 2) if m30.any() else np.nan)
df["txn_count_24h"] = cnt24
df["unique_recipients_7d"] = rec7
df["avg_amount_30d"] = avg30

# KYC notes (account-level) with noise
def kyc(a):
    sus = a["typology"] != "normal"
    if sus and rng.random() < 0.60:
        return rng.choice(KYC_CONCERN)
    if not sus and rng.random() < 0.06:
        return rng.choice(KYC_CONCERN if rng.random() < 0.4 else KYC_MILD)
    if a["segment"] == "student":
        return KYC_BENIGN[6]
    if a["segment"] == "sme":
        return rng.choice([KYC_BENIGN[i] for i in (1, 4, 5, 8, 9)])
    if a["segment"] == "hnw":
        return rng.choice([KYC_BENIGN[i] for i in (8, 13, 1)])
    if a["occupation"] == "Retiree":
        return KYC_BENIGN[7]
    return rng.choice([KYC_BENIGN[i] for i in (0, 1, 2, 3, 4, 8, 10, 11, 12)])


kyc_map = {a["account_id"]: kyc(a) for a in accounts}
df["kyc_note"] = df.account_id.map(kyc_map)

# vague-memo noise on normal transactions (~8%)
nm = df.index[df.is_suspicious == 0]
pick = rng.choice(nm, size=int(0.08 * len(nm)), replace=False)
df.loc[pick, "transaction_memo"] = [fill(rng.choice(MEMO["vague"]), df.timestamp[i]) for i in pick]

# realistic data-quality issues for EDA (documented): ~1.5% missing income, ~1% missing occupation
miss_acc_inc = rng.choice(list(acc), size=12, replace=False)
miss_acc_occ = rng.choice(list(acc), size=8, replace=False)
df.loc[df.account_id.isin(miss_acc_inc), "declared_monthly_income"] = np.nan
df.loc[df.account_id.isin(miss_acc_occ), "occupation"] = np.nan

# final ordering: chronological transaction IDs
df = df.sort_values("timestamp").reset_index(drop=True)
df.insert(0, "transaction_id", [f"TXN{i:06d}" for i in range(1, len(df) + 1)])
df["timestamp"] = df.timestamp.dt.strftime("%Y-%m-%d %H:%M:%S")
cols = ["transaction_id", "account_id", "timestamp", "amount_myr", "txn_type", "channel", "is_cross_border",
        "dest_country_risk", "counterparty_id", "account_age_days", "txn_count_24h", "unique_recipients_7d",
        "avg_amount_30d", "occupation", "declared_monthly_income", "transaction_memo", "kyc_note", "is_suspicious"]
key = df[["transaction_id", "account_id", "typology"]]
df = df[cols]

if __name__ == "__main__":
    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "raw_transactions.csv", index=False)
    key.to_csv(out / "typology_key.csv", index=False)
    print(df.shape, "suspicious rate:", round(df.is_suspicious.mean(), 4))
    print(key.typology.value_counts())
