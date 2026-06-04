"""
Scout-less CT Scan Planner — Complete Test Suite
Run: venv\\Scripts\\python.exe test_cases.py or python3 test_cases.py
"""

import sys
import os
import numpy as np

# ── Ensure script root directory pathing compatibility ────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ==============================================================================
# 🧠 CORE MODEL & REASONING ENGINES
# ==============================================================================

def simulate_nnLandmark(height_cm, weight_kg):
    """
    Simulates sub-voxel coordinate regression using morphometric scaling parameters.
    Replicated exactly to run standalone without requiring full system interface overhead.
    """
    bmi = weight_kg / ((height_cm / 100) ** 2)
    H = height_cm * 10

    # Indian NIOH-based anthropometric tracking constants
    sternum_z = H * 0.200
    t12_z     = H * 0.482
    pubis_z   = H * 0.771

    # Bounded BMI shift calculation targeting soft-tissue structural displacement
    bmi_shift = float(np.clip((bmi - 22.0) * 1.2, -15.0, 30.0))

    lateral_spread = (bmi - 22) * 0.8
    ap_depth = H * 0.14 + (bmi - 22) * 2.5

    return {
        "Sternum (Manubrium)": {
            "coords": (round(lateral_spread * 0.1, 1),
                       round(-ap_depth * 0.45, 1),
                       round(sternum_z + bmi_shift * 0.5, 1)),
        },
        "T12 Vertebra": {
            "coords": (round(0.0, 1),
                       round(ap_depth * 0.1, 1),
                       round(t12_z + bmi_shift, 1)),
        },
        "Pubic Symphysis": {
            "coords": (round(0.0, 1),
                       round(-ap_depth * 0.3, 1),
                       round(pubis_z + bmi_shift * 1.5, 1)),
        },
    }


def reasoning_agent(scan_type, landmarks, height, weight):
    """
    Evaluates landmarks and assigns physical hardware gantry boundaries with safety buffers.
    """
    bmi = weight / ((height / 100) ** 2)
    sternum_z = landmarks["Sternum (Manubrium)"]["coords"][2]
    t12_z     = landmarks["T12 Vertebra"]["coords"][2]
    pubis_z   = landmarks["Pubic Symphysis"]["coords"][2]
    s = scan_type.lower()

    if "chest" in s and "abdomen" not in s and "cap" not in s and "pelvis" not in s:
        z_start, z_end = sternum_z - 20, t12_z + 5
        protocol_id = "CHEST-001"
        conf = 0.91 - max(0, (bmi - 30) * 0.008)
    elif "abdomen" in s and "pelvis" not in s and "cap" not in s:
        z_start, z_end = t12_z - 15, pubis_z + 10
        protocol_id = "ABD-001"
        conf = 0.88 - max(0, (bmi - 30) * 0.010)
    elif "pelvis" in s and "chest" not in s and "cap" not in s:
        z_start, z_end = pubis_z - 160, pubis_z + 25
        protocol_id = "PELVIS-001"
        conf = 0.85 - max(0, (bmi - 30) * 0.005)
    else:  # CAP — full torso staging protocol
        z_start, z_end = sternum_z - 20, pubis_z + 10
        protocol_id = "CHEST-ABD-001"
        conf = 0.87 - max(0, (bmi - 30) * 0.009)

    conf = float(np.clip(conf, 0.60, 0.97))
    return {
        "protocol_id": protocol_id,
        "z_start_mm": round(z_start, 1),
        "z_end_mm":   round(z_end, 1),
        "confidence": conf,
    }


# ==============================================================================
# 📊 TEST MATRICES AND EVALUATION CRITERIA
# ==============================================================================

EXPECTED_SPANS = {
    "Chest CT":                  (250, 700),
    "Abdomen CT":                (300, 700),
    "Pelvis CT":                 (150, 250),
    "Chest-Abdomen-Pelvis (CAP)":(550, 1350),
}

EXPECTED_CONF = {
    # (bmi_low, bmi_high): (conf_min, conf_max)
    (0,   25): (0.86, 0.97),
    (25,  30): (0.82, 0.92),
    (30,  35): (0.75, 0.89),
    (35,  50): (0.65, 0.85),
    (50, 100): (0.60, 0.78),
}

def expected_conf_range(bmi):
    for (lo, hi), rng in EXPECTED_CONF.items():
        if lo <= bmi < hi:
            return rng
    return (0.60, 0.97)


# ── The Complete Catalog of Human Morphology Testing Rows ───────────────────
ALL_TESTS = [
    # ── Category 1: Normal BMI Baselines ──
    ("CAT1", "Avg Indian Male",          165, 65,  "Chest CT"),
    ("CAT1", "Avg Indian Female",        152, 52,  "Chest CT"),
    ("CAT1", "Avg Indian Female-Abdomen",152, 52,  "Abdomen CT"),
    ("CAT1", "Tall Normal Male",         185, 80,  "Chest-Abdomen-Pelvis (CAP)"),
    ("CAT1", "Short Normal Female",      148, 45,  "Chest CT"),

    # ── Category 2: BMI Edge Cases ──
    ("CAT2", "Underweight",              165, 42,  "Chest CT"),
    ("CAT2", "Lower Normal",             165, 57,  "Chest CT"),
    ("CAT2", "Upper Normal",             170, 72,  "Chest CT"),
    ("CAT2", "Overweight Boundary",      170, 76,  "Chest CT"),
    ("CAT2", "Obese I",                  168, 90,  "Chest CT"),
    ("CAT2", "Obese II",                 165, 110, "Chest CT"),
    ("CAT2", "Obese III",                165, 110, "Chest-Abdomen-Pelvis (CAP)"),
    ("CAT2", "Morbidly Obese",           160, 140, "Abdomen CT"),

    # ── Category 3: All Four Scan Types (same patient benchmark) ──
    ("CAT3", "Scan-Chest",               170, 75,  "Chest CT"),
    ("CAT3", "Scan-Abdomen",             170, 75,  "Abdomen CT"),
    ("CAT3", "Scan-Pelvis",              170, 75,  "Pelvis CT"),
    ("CAT3", "Scan-CAP",                 170, 75,  "Chest-Abdomen-Pelvis (CAP)"),

    # ── Category 4: Height Extremes ──
    ("CAT4", "Min Height",               120, 30,  "Chest CT"),
    ("CAT4", "Short Adult",              145, 50,  "Chest CT"),
    ("CAT4", "Average",                  165, 65,  "Chest CT"),
    ("CAT4", "Tall",                     190, 85,  "Chest CT"),
    ("CAT4", "Max Height",               220, 100, "Chest CT"),

    # ── Category 5: Same BMI Different Composition ──
    ("CAT5", "Short+Heavy BMI35",        155, 85,  "Abdomen CT"),
    ("CAT5", "Tall+Heavy BMI24",         190, 85,  "Abdomen CT"),
    ("CAT5", "Short+Light BMI17",        155, 42,  "Chest CT"),
    ("CAT5", "Tall+Light BMI18",         190, 65,  "Chest CT"),

    # ── Category 6: Boundary / Crash Tests ──
    ("CAT6", "Minimum Inputs",           120, 30,  "Chest CT"),
    ("CAT6", "Maximum Inputs",           220, 250, "Chest-Abdomen-Pelvis (CAP)"),
    ("CAT6", "Square BMI",               160, 160, "Abdomen CT"),
    ("CAT6", "Decimal Precision",        167.3,73.7,"Chest CT"),
    ("CAT6", "Just Above Min",           120.5,30.5,"Chest CT"),

    # ── Category 7: Indian Population Specific ──
    ("CAT7", "Urban Indian Male",        165, 70,  "Chest CT"),
    ("CAT7", "Rural Indian Male",        162, 58,  "Chest CT"),
    ("CAT7", "Urban Indian Female",      152, 58,  "Abdomen CT"),
    ("CAT7", "Elderly Indian",           158, 62,  "Chest-Abdomen-Pelvis (CAP)"),
    ("CAT7", "Young Athlete",            175, 78,  "Chest CT"),
    ("CAT7", "Metabolic Syndrome",       162, 88,  "Abdomen CT"),
]


# ==============================================================================
# 🚀 CORE AUTOMATED RUNNER
# ==============================================================================

def run_tests():
    results = []
    failures = []

    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    WARN = "⚠️  WARN"

    header = (f"{'Cat':<5} {'Label':<26} {'H':>4} {'W':>4} {'BMI':>5} "
              f"{'Z-St':>7} {'Z-End':>7} {'Span':>6} {'Conf':>5}  "
              f"{'SpanChk':<10} {'ConfChk':<10} {'ZOrder':<8}")
    
    print("\n" + "=" * len(header))
    print("  SCOUT-LESS CT PLANNER — COMPLETE ADAPTIVE MODEL SUITE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    prev_cat = ""
    for cat, label, h, w, scan in ALL_TESTS:
        bmi = w / ((h / 100) ** 2)
        lm  = simulate_nnLandmark(h, w)
        r   = reasoning_agent(scan, lm, h, w)

        z_start = r["z_start_mm"]
        z_end   = r["z_end_mm"]
        span    = z_end - z_start
        conf    = r["confidence"]

        # ── Mathematical Range & Sequence Checks ──
        span_lo, span_hi = EXPECTED_SPANS[scan]
        span_ok  = PASS if span_lo <= span <= span_hi else FAIL
        conf_lo, conf_hi = expected_conf_range(bmi)
        conf_ok  = PASS if conf_lo <= conf <= conf_hi else WARN
        zorder   = PASS if z_end > z_start else FAIL

        # Cache failed cases to build localized error reports
        if FAIL in (span_ok, zorder):
            failures.append((cat, label, span_ok, conf_ok, zorder,
                             span, span_lo, span_hi, conf, z_start, z_end))

        if cat != prev_cat:
            print()
        prev_cat = cat

        print(f"{cat:<5} {label:<26} {h:>4.0f} {w:>4.0f} {bmi:>5.1f} "
              f"{z_start:>7.1f} {z_end:>7.1f} {span:>6.1f} {conf:>5.2f}  "
              f"{span_ok:<10} {conf_ok:<10} {zorder:<8}")

        results.append({"pass": FAIL not in (span_ok, zorder)})

    print("\n" + "=" * len(header))
    total  = len(results)
    passed = sum(1 for r in results if r["pass"])
    failed = total - passed

    print(f"\n📊 SUMMARY REPORT: {passed}/{total} profiles passed  |  {failed} failures discovered\n")

    if failures:
        print("🚨 DETAILED FAILURE BREAKDOWNS:")
        for f in failures:
            cat, label, span_ok, conf_ok, zorder, span, slo, shi, conf, zs, ze = f
            print(f"    [{cat}] Profile '{label}':")
            if span_ok == FAIL:
                print(f"         ↳ Error: Computed span {span:.1f}mm broken outside bounds [{slo}–{shi}mm]")
            if zorder == FAIL:
                print(f"         ↳ Error: Spatial inversion occurred! Start ({zs}mm) >= End ({ze}mm)")
    else:
        print("🎉 EXCELLENT: All anatomical configurations cleared constraints.")

    # ── Category 3: Protocol Range Consistency Validation ──
    print("\n📦 CATEGORY 3 — Volumetric Boundary Proportions (Patient: 170cm/75kg):")
    cat3 = [(l, s) for c, l, h, w, s in ALL_TESTS if c == "CAT3"]
    for label, scan in cat3:
        lm = simulate_nnLandmark(170, 75)
        r  = reasoning_agent(scan, lm, 170, 75)
        span = r["z_end_mm"] - r["z_start_mm"]
        print(f"    {scan:<32} Calculated Span: {span:.1f}mm | Hardware Target: {r['protocol_id']}")

    # ── Category 4: Height Expansion Linearity Validation ──
    print("\n📏 CATEGORY 4 — Height Linear Tracking Scaling Checks:")
    cat4 = [(l, h, w) for c, l, h, w, s in ALL_TESTS if c == "CAT4"]
    prev_zs = 0
    linearity_valid = True
    for label, h, w in cat4:
        lm = simulate_nnLandmark(h, w)
        r  = reasoning_agent("Chest CT", lm, h, w)
        zs = r["z_start_mm"]
        trend_arrow = "↑" if zs > prev_zs else ("→" if zs == prev_zs else "↓ [FAILED]")
        print(f"    Target Stature: {h:>3.0f}cm | Z-Axis Hardware Start: {zs:>7.1f}mm | {trend_arrow}")
        if zs < prev_zs and prev_zs > 0:
            linearity_valid = False
        prev_zs = zs
    print(f"    Conclusion: {'✅ Height Linear Progression Confirmed' if linearity_valid else '❌ Linearity Distortion Detected'}")

    # ── Verification of Model Confidence Degradation ──
    print("\n📉 MODEL SECURITY — Monotonic Confidence Shift Evaluation (Chest CT, H=165):")
    bmi_sweeps = [(165, 42), (165, 57), (165, 72), (165, 90), (165, 110), (165, 140)]
    prev_conf = 1.0
    monotonic_valid = True
    for h, w in bmi_sweeps:
        bmi = w / ((h / 100) ** 2)
        lm  = simulate_nnLandmark(h, w)
        r   = reasoning_agent("Chest CT", lm, h, w)
        c   = r["confidence"]
        trend_arrow = "↓" if c < prev_conf else ("→" if c == prev_conf else "↑ [FAILED]")
        print(f"    Calculated BMI: {bmi:>5.1f} | Pipeline Confidence: {c:.3f} | {trend_arrow}")
        if c > prev_conf:
            monotonic_valid = False
        prev_conf = c
    print(f"    Conclusion: {'✅ Confidence Decreases Monotonically as Tissue Noise Increases' if monotonic_valid else '❌ Monotonic Rules Broken'}\n")

    # Enforce standard exit signals based on structural execution accuracy
    if failed > 0 or not linearity_valid or not monotonic_valid:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    run_tests()