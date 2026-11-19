"""
Core computation for the Zero-Order Sugeno Fuzzy model of student academic stress.
MF parameters = optimal values from Differential Evolution (same as the report & notebook).
This file does NOT need the dataset, so the app runs instantly.
"""
import numpy as np
import pandas as pd

# ---------- Input domain ----------
# The model was trained on 5-10 h/day. The app accepts 0-24 h: the left/right shoulder sets
# are extended to the domain edges (full membership), so values outside 5-10 are still handled.
DOMAIN = (0.0, 24.0)
TRAIN_RANGE = (5.0, 10.0)
LO, HI = DOMAIN

# ---------- MF parameters from DE: (a, b, c, d) ----------
MF_SLEEP = {
    "Short":    (LO, LO, 5.6971, 6.2914),          # left shoulder (extended to 0 h)
    "Adequate": (5.6971, 6.2914, 8.0241, 8.8391),  # trapezoid
    "Long":     (8.0241, 8.8391, HI, HI),          # right shoulder (extended to 24 h)
}
MF_STUDY = {
    "Light":  (LO, LO, 5.4688, 6.4558),            # left shoulder (extended to 0 h)
    "Medium": (5.4688, 6.4558, 7.5111, 8.5812),    # trapezoid
    "Heavy":  (7.5111, 8.5812, HI, HI),            # right shoulder (extended to 24 h)
}

# ---------- Rule base: (no, Study, Sleep, Output) ----------
RULE_BASE = [
    (1, "Light",  "Short",    "High"),
    (2, "Light",  "Adequate", "Low"),
    (3, "Light",  "Long",     "Low"),
    (4, "Medium", "Short",    "High"),
    (5, "Medium", "Adequate", "Moderate"),
    (6, "Medium", "Long",     "Moderate"),
    (7, "Heavy",  "Short",    "High"),
    (8, "Heavy",  "Adequate", "High"),
    (9, "Heavy",  "Long",     "High"),
]

# ---------- Consequents & classification thresholds ----------
CONSEQUENT = {"Low": 15, "Moderate": 50, "High": 85}
T1, T2 = 32.5, 67.5          # (15+50)/2 and (50+85)/2
DEFAULT_SCORE = 50.0         # used if no rule fires (never happens in the 0-24 domain)


def trapmf(x, params):
    """Trapezoidal membership function (a,b,c,d); a=b -> left shoulder, c=d -> right shoulder."""
    a, b, c, d = params
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a)
    if c < x < d:
        return (d - x) / (d - c)
    return 0.0


def fuzzify(x, sets):
    """Return a dict of membership degrees of x in each fuzzy set."""
    return {name: trapmf(x, p) for name, p in sets.items()}


def classify(z):
    if z <= T1:
        return "Low"
    if z <= T2:
        return "Moderate"
    return "High"


def infer(sleep, study):
    """Zero-order Sugeno inference. Returns every step of the calculation."""
    mu_sleep = fuzzify(sleep, MF_SLEEP)
    mu_study = fuzzify(study, MF_STUDY)

    rows = []
    for no, study_set, sleep_set, out in RULE_BASE:
        ms, mt = mu_study[study_set], mu_sleep[sleep_set]
        w = min(ms, mt)                      # AND operator = MIN
        z_i = CONSEQUENT[out]
        rows.append({
            "Rule": f"R{no}", "Study": study_set, "Sleep": sleep_set, "Output": out,
            "µ Study": ms, "µ Sleep": mt, "w (firing)": w,
            "z_i": z_i, "w × z_i": w * z_i,
        })
    table = pd.DataFrame(rows)

    numerator = float(table["w × z_i"].sum())
    denominator = float(table["w (firing)"].sum())
    z = numerator / denominator if denominator > 0 else DEFAULT_SCORE

    return {
        "mu_sleep": mu_sleep, "mu_study": mu_study, "table": table,
        "numerator": numerator, "denominator": denominator,
        "z": z, "category": classify(z),
    }
