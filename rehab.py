"""Rehabilitation strategy for a corroded RC member: urgency, interventions and repair material spec.

Rule-based decision support. Each intervention is listed with the EN 1504-9 principle and method
it belongs to, the specific findings that triggered it, its main limitations and references.
Roles: PRIMARY = the main remedy for a finding; REQUIRED = must accompany the strategy;
CONTINGENT = adopt if a condition is confirmed or as an alternative.

This is a structured checklist for the engineer, not a design: thresholds marked "judgement" are
the project's assumptions, and every intervention needs specialist design and the current edition
of the cited standard.
"""
import is456
from condition import STATES

GROUPS = ["Structural safety and capacity", "Corrosion control and repair", "Protection and monitoring"]
MIN_CEMENT_KG_M3 = {"Mild": 300, "Moderate": 300, "Severe": 320, "Very severe": 340, "Extreme": 360}  # IS 456 Table 5 (RCC)


def _verdict(checks, name):
    for c in checks:
        if c["check"] == name:
            return c["verdict"]
    return None


def recommend(x, flex, shear, checks, cond):
    p = dict(zip(STATES, cond["posterior"]))
    p_high = p["High"] + p["Severe"]
    p_mod = p["Moderate"] + p_high
    corr = x["corr_pct"]
    hc, ic = x.get("half_cell_mv"), x.get("icorr_ua_cm2")
    cl, carb = x.get("chloride_pct_cement"), x.get("carbonation_mm")
    exp = x.get("exposure_class")
    cover_state, exposed = x.get("cover_condition"), x.get("exposed_rebar")
    water, pattern = x.get("water_ingress"), x.get("crack_pattern")

    flex_v = _verdict(checks, "Flexural capacity vs factored moment")
    shear_v = _verdict(checks, "Shear capacity vs factored shear")
    cover_def = _verdict(checks, "Nominal cover for exposure class") == "deficient"
    crack_exc = _verdict(checks, "Surface crack width") == "exceeded"

    active_evidence = []
    if hc is not None and hc < -350:
        active_evidence.append(f"half-cell {hc:g} mV (< −350 mV, ASTM C876)")
    if ic is not None and ic > 0.5:
        active_evidence.append(f"corrosion rate {ic:g} µA/cm² (> 0.5)")
    if p_high >= 0.5:
        active_evidence.append(f"High/Severe state with probability {p_high:.2f}")
    active = bool(active_evidence)
    chloride_measured_high = cl is not None and cl >= 0.4
    chloride_driven = chloride_measured_high or (cl is None and exp in ("Very severe", "Extreme"))
    chloride_why = (f"chloride {cl:g} % by mass of cement at the bar (≥ 0.4 %)" if chloride_measured_high
                    else f"{exp.lower()} exposure (chloride not measured)" if chloride_driven else "")
    carbonated = carb is not None and carb >= x["cover_mm"]
    damaged_cover = cover_state in ("Delaminated (hollow)", "Spalled")
    section_loss_seen = exposed == "Visible section loss"
    low = flex["low_pct"]
    demand_m, demand_v = x.get("demand_moment_knm"), x.get("demand_shear_kn")
    has_evidence = cond["n_instruments"] + cond["n_observations"] + cond["n_measured"] > 0

    # ---------------------------------------------------------------- urgency
    structural, durability = [], []
    if flex_v == "inadequate":
        structural.append(f"Flexural capacity {flex['mu_res_knm']:.1f} kN·m is below the factored demand {demand_m:g} kN·m.")
    elif flex_v == "marginal":
        structural.append(f"The 90 % lower bound of flexural capacity ({flex['mu_low_knm']:.1f} kN·m) is below the "
                          f"factored demand {demand_m:g} kN·m.")
    elif demand_m is None:
        structural.append(f"Residual flexural capacity {flex['residual_pct']:.0f} % (lower bound {low:.0f} %); "
                          "no factored demand entered, so adequacy is not checked.")
    else:
        structural.append(f"Flexural capacity covers the factored demand even at the 90 % lower bound "
                          f"({flex['mu_low_knm']:.1f} ≥ {demand_m:g} kN·m).")
    if shear_v in ("inadequate", "marginal"):
        structural.append(f"Shear capacity {shear['after']['Vu_kn']:.0f} kN is below the factored shear {demand_v:g} kN.")
    durability.append(f"Corrosion state {cond['state']} (probability {cond['state_prob']:.2f}); "
                      f"average mass loss used: {corr:.1f} %.")
    if active:
        durability.append("Corrosion is likely active: " + "; ".join(active_evidence) + ".")
    if chloride_driven:
        durability.append(f"Chloride-induced corrosion is likely: {chloride_why}.")
    if carbonated:
        durability.append(f"Carbonation ({carb:g} mm) has reached the bars (cover {x['cover_mm']:g} mm).")

    if flex_v == "inadequate" or shear_v == "inadequate" or low < 50 or (cover_state == "Spalled" and p["Severe"] >= 0.5):
        level, action = "Immediate", "restrict load or prop the member"
    elif flex_v == "marginal" or shear_v == "marginal" or p_high >= 0.5 or damaged_cover or section_loss_seen:
        level, action = "Urgent", "repair within months"
    elif p_mod >= 0.5 or cover_def or crack_exc or chloride_measured_high or carbonated:
        level, action = "Planned", "repair in the next maintenance cycle"
    else:
        level, action = "Routine", "monitor at routine inspections"
    if not has_evidence:
        action = "carry out a condition survey first (no inspection data entered)"

    # ---------------------------------------------------------------- interventions
    items = []

    def add(group, name, role, principle, purpose, triggers, limits, refs):
        items.append({"group": group, "name": name, "role": role, "principle": principle, "purpose": purpose,
                      "triggered_by": triggers, "limitations": limits, "references": refs})

    G1, G2, G3 = GROUPS
    if level == "Immediate":
        add(G1, "Load restriction or temporary propping", "REQUIRED", "Safety measure (not an EN 1504 method)",
            "Reduce the demand now, so that the rest of the assessment and repair is not done under pressure.",
            structural[:1] + ([f"Lower-bound residual strength {low:.0f} % (< 50 %, judgement)"] if low < 50 else []),
            ["Temporary: it does not stop corrosion or restore capacity.",
             "Props must be designed for the full load path down to a sound support."],
            ["ACI 364.1R (assessment before rehabilitation)"])

    strengthen_triggers = []
    if flex_v == "inadequate":
        strengthen_triggers.append("flexural capacity below the factored demand")
    if flex_v == "marginal":
        strengthen_triggers.append("flexural lower bound below the factored demand")
    if demand_m is None and flex["residual_pct"] < 80:
        strengthen_triggers.append(f"residual strength {flex['residual_pct']:.0f} % (< 80 %, judgement) with no "
                                   "factored demand entered - check adequacy first")
    if strengthen_triggers:
        add(G1, "Flexural strengthening (externally bonded FRP or steel plates)",
            "PRIMARY" if flex_v == "inadequate" else "CONTINGENT", "EN 1504-9 Principle 4 (SS), method 4.3",
            "Restore flexural capacity by bonding tension reinforcement to the soffit.",
            [s[0].upper() + s[1:] for s in strengthen_triggers],
            ["Corrosion must be stopped and the substrate repaired first; FRP over active corrosion hides the damage.",
             "Needs a sound, prepared substrate (pull-off testing) and fire protection where required.",
             "Section enlargement with added bars (methods 4.1 and 4.4) is the alternative where FRP is unsuitable."],
            ["ACI 440.2R (externally bonded FRP)", "EN 1504-9"])
    if shear_v in ("inadequate", "marginal"):
        add(G1, "Shear strengthening (FRP U-wraps or added stirrups with jacketing)",
            "PRIMARY" if shear_v == "inadequate" else "CONTINGENT", "EN 1504-9 Principle 4 (SS), methods 4.3 / 4.4",
            "Restore shear capacity lost with corroded stirrups.",
            [f"Shear capacity {shear['after']['Vu_kn']:.0f} kN vs factored shear {demand_v:g} kN ({shear_v})"],
            ["Stirrups usually corrode first (small diameter, least cover); confirm their condition by break-out.",
             "U-wraps need anchorage at the top of the web."],
            ["ACI 440.2R", "EN 1504-9"])
    if section_loss_seen or corr >= 15:
        why = []
        if section_loss_seen:
            why.append("visible section loss of exposed bars")
        if corr >= 15:
            why.append(f"average mass loss {corr:.1f} % (≥ 15 %, judgement)")
        add(G1, "Supplementary or replacement reinforcement during repair",
            "REQUIRED" if section_loss_seen else "CONTINGENT", "EN 1504-9 Principle 4 (SS), method 4.1",
            "Replace the lost steel area where bars are exposed during patch repair.",
            [w[0].upper() + w[1:] for w in why],
            ["Measure the remaining diameter at the worst pits, not only the average.",
             "Lap new bars with the full lap length (IS 456 cl. 26.2.5)."],
            ["EN 1504-9", "IS 456:2000 cl. 26.2.5"])
    if crack_exc and pattern != "Longitudinal along bars":
        add(G1, "Structural crack injection (epoxy or cementitious)", "CONTINGENT", "EN 1504-9 Principle 4 (SS), method 4.5",
            "Restore monolithic action across cracks and close the path for water and chlorides.",
            [f"Crack width {x['crack_width_mm']:g} mm exceeds the IS 456 cl. 35.3.2 limit"],
            ["Only for dormant cracks; live cracks need a flexible filler or a joint (method 1.6).",
             "Do not inject cracks caused by corroding bars without repairing the corrosion."],
            ["EN 1504-5 (concrete injection)", "ACI 224.1R (repair of cracks)"])
    if p["Severe"] >= 0.5 or low < 50:
        add(G1, "Member replacement", "CONTINGENT", "EN 1504-9 Principle 3 (CR), method 3.4",
            "Replace the element if strengthening and repair would be uneconomic or unreliable.",
            [f"Severe state probability {p['Severe']:.2f}" if p["Severe"] >= 0.5 else
             f"Lower-bound residual strength {low:.0f} %"],
            ["Needs temporary works and a check of the connections to adjacent members."],
            ["EN 1504-9"])

    patch = damaged_cover or section_loss_seen or exposed == "Pitting" or p_high >= 0.5
    if patch or chloride_measured_high or carbonated:
        why = []
        if damaged_cover:
            why.append(f"cover {cover_state.lower()}")
        if exposed in ("Pitting", "Visible section loss"):
            why.append(f"exposed bars: {exposed.lower()}")
        if p_high >= 0.5:
            why.append(f"High/Severe corrosion probability {p_high:.2f}")
        if chloride_measured_high:
            why.append(f"chloride {cl:g} % at the bar")
        if carbonated:
            why.append("carbonation front at the bar")
        add(G2, "Patch repair: remove contaminated concrete, clean bars, reinstate cover",
            "PRIMARY" if patch else "CONTINGENT", "EN 1504-9 Principles 3 (CR) and 7 (RP), methods 3.1 / 3.3 and 7.2",
            "Remove damaged, carbonated or chloride-contaminated concrete around the bars and rebuild with repair mortar.",
            [w[0].upper() + w[1:] for w in why],
            ["Break out behind the bars and beyond the visibly damaged area.",
             "In chloride-contaminated concrete, patches alone shift corrosion to their edges (incipient anodes).",
             "Use a structural repair mortar (EN 1504-3 class R4, R3 minimum)."],
            ["EN 1504-3 (repair mortars)", "EN 1504-10 (site application)", "ACI 546R (concrete repair)"])
        iccp_primary = chloride_driven and active and p_high >= 0.5
        if chloride_driven and patch:
            add(G2, "Discrete galvanic (sacrificial) anodes at patch perimeters",
                "CONTINGENT" if iccp_primary else "REQUIRED",
                "EN 1504-9 Principle 10 (CP), method 10.1 (galvanic)",
                "Protect the steel around each patch from the incipient-anode effect.",
                [f"Patch repair in chloride conditions: {chloride_why}"]
                + (["Alternative to impressed current CP where that is not adopted"] if iccp_primary else []),
                ["Anodes are consumed over time and protect only the zone near the patch.",
                 "Needs electrical continuity between anode and reinforcement."],
                ["EN ISO 12696 (cathodic protection of steel in concrete)"])
    if chloride_driven and active:
        add(G2, "Impressed current cathodic protection", "PRIMARY" if p_high >= 0.5 else "CONTINGENT",
            "EN 1504-9 Principle 10 (CP), method 10.1",
            "Hold the steel at a potential where corrosion cannot proceed, without removing all contaminated concrete.",
            [f"Chloride-induced: {chloride_why}", "Active corrosion: " + "; ".join(active_evidence)],
            ["Needs continuity of the reinforcement, a permanent power supply and monitoring.",
             "Specialist design; take care with prestressed or high-strength steel (hydrogen embrittlement)."],
            ["EN ISO 12696", "ACI 222R"])
    if chloride_measured_high and p_high < 0.5:
        add(G2, "Electrochemical chloride extraction", "CONTINGENT", "EN 1504-9 Principle 7 (RP), method 7.5",
            "Draw chlorides out of the cover over several weeks while corrosion is still limited.",
            [f"Chloride {cl:g} % at the bar, but High/Severe probability only {p_high:.2f}"],
            ["Chlorides re-enter unless the surface is protected afterwards.",
             "Not for prestressed members or alkali-reactive aggregates."],
            ["CEN/TS 14038-2 (chloride extraction)"])
    if carbonated and not chloride_measured_high and p_high < 0.5:
        add(G2, "Electrochemical realkalisation", "CONTINGENT", "EN 1504-9 Principle 7 (RP), method 7.3",
            "Restore alkalinity around the bars so the steel re-passivates.",
            [f"Carbonation {carb:g} mm ≥ cover {x['cover_mm']:g} mm, chloride not high"],
            ["Not for chloride-induced corrosion.", "Not for prestressed members or alkali-reactive aggregates."],
            ["EN 14038-1 (realkalisation)"])
    if has_evidence and 0.3 <= p_mod and p_high < 0.5:
        add(G2, "Migrating corrosion inhibitor", "CONTINGENT", "EN 1504-9 Principle 11 (CA), method 11.3",
            "Slow corrosion of bars that are depassivated but not yet heavily corroded.",
            [f"Moderate-or-worse probability {p_mod:.2f}, High/Severe {p_high:.2f}"],
            ["Effectiveness is uncertain at high chloride levels; confirm by corrosion-rate monitoring."],
            ["EN 1504-9"])
    if cover_def:
        req, _ = is456.required_cover_mm(exp, x["fck_mpa"], x["bar_dia_mm"])
        add(G2, "Increase cover with an overlay of dense mortar or concrete",
            "PRIMARY" if exp in ("Severe", "Very severe", "Extreme") else "CONTINGENT",
            "EN 1504-9 Principle 7 (RP), method 7.1",
            "Bring the cover up to the IS 456 requirement so the remaining life is governed by a compliant section.",
            [f"Cover {x['cover_mm']:g} mm is below the {req} mm required for {exp.lower()} exposure (IS 456 Table 16)"],
            ["Adds dead load and changes the section; check headroom and clearances.",
             "Needs good bond to a prepared substrate."],
            ["IS 456:2000 Table 16", "EN 1504-3"])

    if exp in ("Severe", "Very severe", "Extreme") or cover_def or carbonated or chloride_measured_high:
        why = [f"{exp} exposure" if exp else None, "deficient cover" if cover_def else None,
               "carbonation at the bar" if carbonated else None, "chloride at the bar" if chloride_measured_high else None]
        add(G3, "Anti-carbonation / chloride-barrier coating", "PRIMARY", "EN 1504-9 Principles 1 (PI) and 8 (IR), methods 1.3 / 8.3",
            "Slow the ingress of carbon dioxide, chlorides and water that sustain corrosion.",
            [w[0].upper() + w[1:] for w in why if w],
            ["Does not stop corrosion that is already active; use with repair.",
             "Must be vapour-permeable and re-applied over the structure's life."],
            ["EN 1504-2 (surface protection)"])
    if water in ("Occasional", "Persistent"):
        add(G3, "Correct drainage, joints and water paths", "REQUIRED" if water == "Persistent" else "CONTINGENT",
            "Good practice: moisture control at source",
            "Remove the water supply that drives the corrosion cell - often the cheapest intervention.",
            [f"Water ingress: {water.lower()}"],
            ["Find the source (leaking joints, ponding, failed waterproofing) before repairing."],
            ["ACI 546R"])
        add(G3, "Hydrophobic impregnation (silane / siloxane)", "CONTINGENT", "EN 1504-9 Principles 1, 2 and 8, methods 1.1 / 2.1 / 8.1",
            "Reduce water and chloride uptake while letting the concrete dry out.",
            [f"Water ingress: {water.lower()}"],
            ["Not effective against water under pressure; re-apply periodically."],
            ["EN 1504-2"])
    if not has_evidence:
        add(G3, "Condition survey before any repair decision", "REQUIRED", "Assessment (not an EN 1504 method)",
            "Half-cell and resistivity mapping, cover meter, chloride and carbonation tests, and a visual and "
            "hammer-tapping survey - the corrosion state is otherwise only a prior guess.",
            ["No NDT, visual or corrosion measurement entered"],
            ["Choose test locations where exposure and visible damage are worst, plus a sound reference area."],
            ["ACI 228.2R (NDT methods)", "ASTM C876"])
    interval = {"Immediate": "monthly until repaired", "Urgent": "every 6 months", "Planned": "every 12 months",
                "Routine": "at routine inspections (2-5 years)"}[level]
    if has_evidence:
        add(G3, "Corrosion monitoring", "REQUIRED" if p_mod >= 0.3 or level != "Routine" else "CONTINGENT",
            "Assessment and monitoring (not an EN 1504 method)",
            f"Half-cell and resistivity mapping, crack and spall survey - suggested {interval}; after repair, to "
            "confirm it works.",
            [f"Urgency: {level}", f"Moderate-or-worse probability {p_mod:.2f}"],
            ["Intervals are suggestions (judgement); adjust to the consequence of failure."],
            ["ASTM C876", "ACI 228.2R (NDT methods)"])

    spec = None
    if exp:
        req, notes = is456.required_cover_mm(exp, x["fck_mpa"], x["bar_dia_mm"])
        spec = {"exposure_class": exp, "min_grade": f"M{is456.MIN_GRADE_MPA[exp]}",
                "max_wc": is456.MAX_WC_RATIO[exp], "min_cement_kg_m3": MIN_CEMENT_KG_M3[exp], "nominal_cover_mm": req,
                "repair_mortar": "EN 1504-3 class R4 (structural); R3 minimum",
                "note": "IS 456:2000 Tables 5 and 16 for reinforced concrete in this exposure class."}

    order = {"REQUIRED": 0, "PRIMARY": 1, "CONTINGENT": 2}
    items.sort(key=lambda it: (GROUPS.index(it["group"]), order[it["role"]]))
    return {"urgency": {"level": level, "action": action, "structural": structural, "durability": durability},
            "interventions": items, "material_spec": spec}
