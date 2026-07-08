"""
feature_definitions.py

Defines the 23 flow-level features used by FLAIR for the WUSTL-IIoT dataset,
matching Table III ("Selected Traffic Features") from the paper.

IMPORTANT:
- These are the 23 "selected" features (not all 41 columns in the CSV).
- Column names here must match the CSV header exactly.
"""

# -------------------------------
# 23 Selected Features (Table III)
# -------------------------------

# Ports are treated as categorical identifiers (we can embed them or one-hot later).
CATEGORICAL_FEATURES = [
    "Sport",
    "Dport",
    "Proto",
]

# Everything else is numeric.
NUMERIC_FEATURES = [
    # Mean flow duration (mean)
    "Mean",

    # Packet counts
    "SrcPkts",
    "DstPkts",
    "TotPkts",

    # Byte counts
    "SrcBytes",
    "DstBytes",
    "TotBytes",

    # Loads (bits/sec)
    "SrcLoad",
    "DstLoad",
    "Load",

    # Rates (pkts/sec)
    "SrcRate",
    "DstRate",
    "Rate",

    # Loss
    "SrcLoss",
    "DstLoss",
    "Loss",
    "pLoss",

    # Jitter (ms)
    "SrcJitter",
    "DstJitter",

    # Inter-packet arrival (ms)
    "SIntPkt",
    "DIntPkt",
]

# The exact 24 features the model consumes (3 categorical + 21 numeric)
ALL_FEATURE_NAMES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Optional: keep a dict for descriptions (nice for thesis figures/logging)
FLOW_FEATURES = {
    "Mean": "Average duration of active flows",
    "Sport": "Source port number",
    "Dport": "Destination port number",
    "Proto": "Transport layer protocol (e.g., TCP, UDP)",
    "SrcPkts": "Source→Destination packet count",
    "DstPkts": "Destination→Source packet count",
    "TotPkts": "Total transaction packet count",
    "SrcBytes": "Source→Destination byte count",
    "DstBytes": "Destination→Source byte count",
    "TotBytes": "Total transaction byte count",
    "SrcLoad": "Source bits per second",
    "DstLoad": "Destination bits per second",
    "Load": "Total bits per second",
    "SrcRate": "Source packets per second",
    "DstRate": "Destination packets per second",
    "Rate": "Total packets per second",
    "SrcLoss": "Source packets retransmitted/dropped",
    "DstLoss": "Destination packets retransmitted/dropped",
    "Loss": "Total packets retransmitted/dropped",
    "pLoss": "Percent packets retransmitted/dropped",
    "SrcJitter": "Source jitter (ms)",
    "DstJitter": "Destination jitter (ms)",
    "SIntPkt": "Source interpacket arrival time (ms)",
    "DIntPkt": "Destination interpacket arrival time (ms)",
}