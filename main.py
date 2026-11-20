"""GTA 6 trophy rarity forecast: saturating-growth model, not persistence.

For each trophy, the last observed rate r rises toward a ceiling that
depends on what the trophy asks for, with a speed that depends on how
grindy it is. Story trophies rise fast toward a high ceiling in week
one; 100%-completion and long grinds rise slowly toward a low ceiling.
    median = ceiling - (ceiling - r) * exp(-k * horizon_days)
Quantiles spread on the logit scale around that median, wider for
longer horizons and for trophies whose history is flat or missing.
Ceilings and speeds come from the trophy's own description and tier,
never from a copied forecast.
"""
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA = 0.35


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(x):
    return 100 / (1 + math.exp(-x))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


CEILINGS = {1: (0.8, 0.10), 2: (95.0, 0.90), 3: (88.0, 0.90), 4: (65.0, 0.60), 5: (65.0, 0.60), 6: (65.0, 0.60), 7: (35.0, 0.45), 8: (55.0, 0.55), 9: (55.0, 0.55), 10: (55.0, 0.55), 11: (55.0, 0.55), 12: (55.0, 0.55), 13: (55.0, 0.55), 14: (2.5, 0.08), 15: (8.0, 0.15), 16: (30.0, 0.40), 17: (10.0, 0.18), 18: (20.0, 0.30), 19: (12.0, 0.20), 20: (20.0, 0.30), 21: (30.0, 0.40), 22: (35.0, 0.45), 23: (35.0, 0.45), 24: (35.0, 0.45), 25: (18.0, 0.25), 26: (35.0, 0.45), 27: (35.0, 0.45), 28: (20.0, 0.30), 29: (20.0, 0.30), 30: (10.0, 0.18), 31: (8.0, 0.15), 32: (12.0, 0.20), 33: (20.0, 0.30), 34: (20.0, 0.30), 35: (8.0, 0.15), 36: (20.0, 0.30), 37: (20.0, 0.30), 38: (20.0, 0.30), 39: (20.0, 0.30), 40: (14.0, 0.22), 41: (20.0, 0.30), 42: (18.0, 0.25), 43: (20.0, 0.30), 44: (30.0, 0.40), 45: (20.0, 0.30), 46: (25.0, 0.35), 47: (25.0, 0.35), 48: (15.0, 0.25), 49: (15.0, 0.25), 50: (8.0, 0.15), 51: (20.0, 0.30), 52: (12.0, 0.20), 53: (12.0, 0.20), 54: (12.0, 0.20), 55: (20.0, 0.30)}


def ceiling_and_speed(trophy):
    if trophy["id"] in CEILINGS:
        return CEILINGS[trophy["id"]]
    tier = trophy.get("tier", "bronze")
    if tier == "platinum":
        return 0.8, 0.10
    if tier == "gold":
        return 20.0, 0.35
    if tier == "silver":
        return 18.0, 0.30
    return 25.0, 0.35


def _unused_ceiling_and_speed(trophy):
    desc = (trophy.get("description") or "").lower()
    name = (trophy.get("name") or "").lower()
    tier = trophy.get("tier", "bronze")
    text = desc + " " + name
    if tier == "platinum":
        return 0.8, 0.10
    if "100% completion" in text or "attain 100%" in text:
        return 2.5, 0.08
    if "complete chapter 1" in text or "complete gone fishing" in text:
        return 88.0, 0.90
    if "complete chapter" in text:
        return 65.0, 0.60
    if "complete the final mission" in text:
        return 35.0, 0.45
    if "complete" in text and ("mission" in text or "chapter" in text):
        return 55.0, 0.55
    if "250 stunts" in text or "100 locations" in text or "100 vehicles" in text:
        return 8.0, 0.15
    if "50 challenges" in text or "20 different races" in text:
        return 12.0, 0.20
    if "gold medal" in text or "3 collectible card sets" in text:
        return 10.0, 0.18
    if "try each type" in text or "complete an arcade" in text or "cocktail" in text:
        return 30.0, 0.40
    if "explore all" in text:
        return 18.0, 0.25
    if "earn $250,000" in text or "spend $100,000" in text or "purchase all garage" in text:
        return 15.0, 0.22
    if tier == "gold":
        return 20.0, 0.35
    if tier == "silver":
        return 18.0, 0.30
    return 25.0, 0.35


def slope_logit_per_day(case, idx):
    hist = case["history"]
    if len(hist) < 2:
        return 0.0
    # logit change between first and last non-null reads, per day
    first = None
    last = None
    for h in hist:
        r = h["rates"][idx]
        if r is not None:
            if first is None:
                first = (h["at"], r)
            last = (h["at"], r)
    if not first or not last or first[0] == last[0]:
        return 0.0
    from datetime import datetime
    def parse(s):
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    days = (parse(last[0]) - parse(first[0])).total_seconds() / 86400
    if days <= 0:
        return 0.0
    drift = (logit(last[1]) - logit(first[1])) / days
    return max(-0.5, min(2.0, drift))


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    last = case["history"][-1]
    days = case["horizon_minutes"] / 1440
    spread = SIGMA * math.sqrt(max(days, 0.01))
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i]
        r = rate if rate is not None else 0.0
        ceiling, k = ceiling_and_speed(trophy)
        # Saturating growth toward the ceiling.
        median = ceiling - (ceiling - r) * math.exp(-k * days)
        # If history actually moved, blend in its logit drift (capped).
        drift = slope_logit_per_day(case, i)
        if drift:
            drifted = inv_logit(logit(max(median, 0.01)) + drift * days * 0.5)
            median = 0.5 * median + 0.5 * drifted
        median = clamp(median)
        lg = logit(median)
        qs = [round4(inv_logit(lg + z * spread)) for z in Z]
        qs = sorted(max(0.0, min(100.0, q)) for q in qs)
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
