import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import datetime
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="🚛",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #4361ee;
        margin-bottom: 1rem;
    }
    .metric-card.urgent   { border-left-color: #e63946; background: #fff5f5; }
    .metric-card.tyre     { border-left-color: #f4a261; background: #fffbf0; }
    .metric-card.preventive { border-left-color: #2a9d8f; background: #f0faf9; }
    .metric-card.null-type  { border-left-color: #adb5bd; background: #f8f9fa; }
    .card-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6c757d;
        margin-bottom: 0.2rem;
    }
    .card-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a2e;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    .stButton > button {
        background-color: #4361ee;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover { background-color: #3a0ca3; color: white; }
    .alert-urgent {
        background: #fff5f5;
        border: 1px solid #e63946;
        border-radius: 8px;
        padding: 1rem;
        color: #c1121f;
        font-weight: 600;
    }
    .alert-ok {
        background: #f0faf9;
        border: 1px solid #2a9d8f;
        border-radius: 8px;
        padding: 1rem;
        color: #1a7a6e;
        font-weight: 600;
    }
    /* Gauge */
    .gauge-wrapper {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.4rem 1.5rem 1.2rem;
        border-left: 4px solid #e63946;
        margin-bottom: 1rem;
    }
    .gauge-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6c757d;
        margin-bottom: 0.6rem;
    }
    .gauge-track {
        background: #e9ecef;
        border-radius: 999px;
        height: 18px;
        width: 100%;
        overflow: hidden;
    }
    .gauge-fill { height: 100%; border-radius: 999px; transition: width 0.6s ease; }
    .gauge-pct  { font-size: 1.6rem; font-weight: 800; margin-top: 0.5rem; }
    .gauge-desc { font-size: 0.82rem; color: #6c757d; margin-top: 0.25rem; }
    /* Fleet summary cards */
    .fleet-summary-card {
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .fleet-summary-card.all    { background: #eef2ff; border: 2px solid #4361ee; }
    .fleet-summary-card.urgent { background: #fff5f5; border: 2px solid #e63946; }
    .fleet-summary-card.tyre   { background: #fffbf0; border: 2px solid #f4a261; }
    .fleet-summary-card.prev   { background: #f0faf9; border: 2px solid #2a9d8f; }
    .fleet-summary-card.none   { background: #f8f9fa; border: 2px solid #adb5bd; }
    .fleet-card-num { font-size: 2.2rem; font-weight: 800; line-height: 1.1; }
    .fleet-card-num.all    { color: #4361ee; }
    .fleet-card-num.urgent { color: #e63946; }
    .fleet-card-num.tyre   { color: #f4a261; }
    .fleet-card-num.prev   { color: #2a9d8f; }
    .fleet-card-num.none   { color: #adb5bd; }
    .fleet-card-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6c757d;
        margin-top: 0.3rem;
    }
    .fleet-card-icon { font-size: 1.5rem; margin-bottom: 0.3rem; }
    /* Notes box */
    .notes-box {
        background: #fffde7;
        border: 1px solid #f9c74f;
        border-radius: 10px;
        padding: 1rem 1.3rem;
        margin: 1rem 0;
        font-size: 0.92rem;
        color: #5a4000;
    }
    .notes-box ul { margin: 0.4rem 0 0 1.2rem; padding: 0; }
    .notes-box li { margin-bottom: 0.25rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "fleet"
if "selected_vehicle" not in st.session_state:
    st.session_state.selected_vehicle = None

# ─────────────────────────────────────────────
# LABEL / PRIORITY MAPS
# ─────────────────────────────────────────────
LABEL_MAP = {
    0: ("Preventive Maintenance",  "preventive", "🟢"),
    1: ("Tyre Maintenance",        "tyre",        "🟡"),
    2: ("Corrective Maintenance",  "urgent",      "🔴"),
    3: ("No Maintenance Record",   "null-type",   "⚪"),
}
PRIORITY_MAP = {0: "Low", 1: "Medium", 2: "Critical", 3: "Unknown"}

MAINT_FILTER_OPTIONS = [
    "All Vehicles",
    "Corrective Maintenance (Urgent)",
    "Tyre Maintenance",
    "Preventive Maintenance",
    "No Maintenance Record",
]
# Map label → pred_class integer (no emojis to avoid encoding issues)
MAINT_FILTER_MAP = {
    "Corrective Maintenance (Urgent)": 2,
    "Tyre Maintenance":                1,
    "Preventive Maintenance":          0,
    "No Maintenance Record":           3,
}

# ─────────────────────────────────────────────
# DB CONNECTION
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_engine():
    return create_engine(
        "mssql+pyodbc://sa:#saProdata123!@thinkpad-x230-linux/vms?"
        "driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
    )

# ─────────────────────────────────────────────
# LOAD & TRAIN MODEL
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_train():
    engine = get_engine()
    query = """
    SELECT
        b.[EquipmentID], a.[Priority], a.DateReceived, d.[last_service_date],
        a.Jarak_Operasi, a.vehicle_category, a.maintenance_type,
        b.breakdown_count_30d, b.breakdown_count_90d,
        c.odometer_maintenance, c.odometer_repair, c.odometer_tyre,
        d.[service_count], e.OdometerLastUpdate,
        e.[CurrentOdometer], e.[NextServiceKm]
    FROM [vms].[dbo].[vw_service_request_clean] a
    JOIN [vms].[dbo].[vw_vehicle_breakdown_stats] b ON a.vh_regno = b.[EquipmentID]
    JOIN [vms].[dbo].[vw_vehicle_odometer] c ON a.vh_regno = c.[EquipmentID]
    JOIN [vms].[dbo].[vw_ui_vehicle_service_summary] d
        ON a.vh_regno = d.vehicle_id AND a.DateReceived = c.DateReceived
    JOIN [vms].[dbo].[vw_ui_next_service_km] e ON a.vh_regno = e.[VehicleRegNo]
    WHERE a.DateReceived BETWEEN '2021-01-01' AND '{today}'
      AND d.[last_service_date] BETWEEN '2021-01-01' AND '{today}'
      AND e.OdometerLastUpdate  BETWEEN '2021-01-01' AND '{today}'
    """.format(today=datetime.date.today().strftime('%Y-%m-%d'))
    data = pd.read_sql(query, engine)

    for col_date, prefix in [
        ("last_service_date", "lastservice"),
        ("DateReceived",      "datereceived"),
        ("OdometerLastUpdate","odometer"),
    ]:
        data[col_date] = pd.to_datetime(data[col_date])
        data[f"year_{prefix}"]  = data[col_date].dt.year
        data[f"month_{prefix}"] = data[col_date].dt.month
        data[f"day_{prefix}"]   = data[col_date].dt.day
    data = data.drop(columns=["last_service_date", "OdometerLastUpdate"])

    for col in ["Priority","Jarak_Operasi","breakdown_count_30d","breakdown_count_90d",
                "odometer_maintenance","odometer_repair","odometer_tyre",
                "service_count","CurrentOdometer","NextServiceKm"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)

    data = data.drop_duplicates()

    urgency_map = {"Preventive Maintenance ": 0, "Tyre": 1, "Corrective Maintenance": 2}
    data["urgent_maintenance"] = data["maintenance_type"].map(urgency_map).fillna(3).astype(int)
    data["maintenance_type"]   = data["maintenance_type"].fillna("Unknown / No Record")
    data["vehicle_category"]   = data["vehicle_category"].fillna("Unknown")

    le_vehicle = LabelEncoder()
    le_maint   = LabelEncoder()
    le_equip   = LabelEncoder()
    data["vehicle_category_enc"] = le_vehicle.fit_transform(data["vehicle_category"])
    data["maintenance_type_enc"] = le_maint.fit_transform(data["maintenance_type"])
    data["EquipmentID_enc"]      = le_equip.fit_transform(data["EquipmentID"])

    feature_cols = [
        "Priority","Jarak_Operasi","breakdown_count_30d","breakdown_count_90d",
        "odometer_maintenance","odometer_repair","odometer_tyre",
        "service_count","CurrentOdometer","NextServiceKm",
        "vehicle_category_enc","maintenance_type_enc","EquipmentID_enc",
        "year_lastservice","month_lastservice","day_lastservice",
        "year_datereceived","month_datereceived","day_datereceived",
        "year_odometer","month_odometer","day_odometer",
    ]

    X = data[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = data["urgent_maintenance"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=200, max_depth=10, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.8, random_state=42,
        eval_metric="mlogloss", use_label_encoder=False,
    )
    model.fit(X_train_sm, y_train_sm)

    equipment_ids = sorted(data["EquipmentID"].dropna().unique().tolist())
    return model, le_vehicle, le_maint, le_equip, feature_cols, data, equipment_ids

# ─────────────────────────────────────────────
# FETCH ALL VEHICLES (latest record per vehicle)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=300)
def fetch_all_vehicles():
    engine = get_engine()
    query = """
    SELECT
        b.[EquipmentID], a.[Priority], a.DateReceived, d.[last_service_date],
        a.Jarak_Operasi, a.vehicle_category, a.maintenance_type,
        b.breakdown_count_30d, b.breakdown_count_90d,
        c.odometer_maintenance, c.odometer_repair, c.odometer_tyre,
        d.[service_count], e.OdometerLastUpdate,
        e.[CurrentOdometer], e.[NextServiceKm]
    FROM [vms].[dbo].[vw_service_request_clean] a
    JOIN [vms].[dbo].[vw_vehicle_breakdown_stats] b ON a.vh_regno = b.[EquipmentID]
    JOIN [vms].[dbo].[vw_vehicle_odometer] c ON a.vh_regno = c.[EquipmentID]
    JOIN [vms].[dbo].[vw_ui_vehicle_service_summary] d
        ON a.vh_regno = d.vehicle_id AND a.DateReceived = c.DateReceived
    JOIN [vms].[dbo].[vw_ui_next_service_km] e ON a.vh_regno = e.[VehicleRegNo]
    WHERE a.DateReceived = (
        SELECT MAX(a2.DateReceived)
        FROM [vms].[dbo].[vw_service_request_clean] a2
        WHERE a2.vh_regno = a.vh_regno
    )
    """
    return pd.read_sql(query, engine)

# ─────────────────────────────────────────────
# FETCH SINGLE VEHICLE
# ─────────────────────────────────────────────
def fetch_vehicle(equipment_id: str):
    engine = get_engine()
    query = f"""
    SELECT TOP 1
        b.[EquipmentID], a.[Priority], a.DateReceived, d.[last_service_date],
        a.Jarak_Operasi, a.vehicle_category, a.maintenance_type,
        b.breakdown_count_30d, b.breakdown_count_90d,
        c.odometer_maintenance, c.odometer_repair, c.odometer_tyre,
        d.[service_count], e.OdometerLastUpdate,
        e.[CurrentOdometer], e.[NextServiceKm]
    FROM [vms].[dbo].[vw_service_request_clean] a
    JOIN [vms].[dbo].[vw_vehicle_breakdown_stats] b ON a.vh_regno = b.[EquipmentID]
    JOIN [vms].[dbo].[vw_vehicle_odometer] c ON a.vh_regno = c.[EquipmentID]
    JOIN [vms].[dbo].[vw_ui_vehicle_service_summary] d
        ON a.vh_regno = d.vehicle_id AND a.DateReceived = c.DateReceived
    JOIN [vms].[dbo].[vw_ui_next_service_km] e ON a.vh_regno = e.[VehicleRegNo]
    WHERE b.[EquipmentID] = '{equipment_id}'
    ORDER BY a.DateReceived DESC
    """
    return pd.read_sql(query, engine)

# ─────────────────────────────────────────────
# PREDICT ROW
# ─────────────────────────────────────────────
def predict_row(row, model, le_vehicle, le_maint, le_equip, feature_cols):
    def safe_enc(le, val):
        try:
            return le.transform([val])[0]
        except ValueError:
            return 0

    for _col, _fill in [("maintenance_type", "Unknown / No Record"), ("vehicle_category", "Unknown")]:
        if pd.isna(row.get(_col)):
            row[_col] = _fill

    _today = datetime.date.today()
    feat = {
        "Priority":             pd.to_numeric(row.get("Priority", 0), errors="coerce") or 0,
        "Jarak_Operasi":        pd.to_numeric(row.get("Jarak_Operasi", 0), errors="coerce") or 0,
        "breakdown_count_30d":  pd.to_numeric(row.get("breakdown_count_30d", 0), errors="coerce") or 0,
        "breakdown_count_90d":  pd.to_numeric(row.get("breakdown_count_90d", 0), errors="coerce") or 0,
        "odometer_maintenance": pd.to_numeric(row.get("odometer_maintenance", 0), errors="coerce") or 0,
        "odometer_repair":      pd.to_numeric(row.get("odometer_repair", 0), errors="coerce") or 0,
        "odometer_tyre":        pd.to_numeric(row.get("odometer_tyre", 0), errors="coerce") or 0,
        "service_count":        pd.to_numeric(row.get("service_count", 0), errors="coerce") or 0,
        "CurrentOdometer":      pd.to_numeric(row.get("CurrentOdometer", 0), errors="coerce") or 0,
        "NextServiceKm":        pd.to_numeric(row.get("NextServiceKm", 0), errors="coerce") or 0,
        "vehicle_category_enc": safe_enc(le_vehicle, str(row.get("vehicle_category", ""))),
        "maintenance_type_enc": safe_enc(le_maint,   str(row.get("maintenance_type", ""))),
        "EquipmentID_enc":      safe_enc(le_equip,   str(row.get("EquipmentID", ""))),
    }
    for col_name, prefix, src in [
        ("last_service_date",  "lastservice",   row.get("last_service_date", pd.NaT)),
        ("DateReceived",       "datereceived",  _today),
        ("OdometerLastUpdate", "odometer",      row.get("OdometerLastUpdate", pd.NaT)),
    ]:
        val = pd.to_datetime(src, errors="coerce") if not isinstance(src, datetime.date) else pd.Timestamp(src)
        feat[f"year_{prefix}"]  = val.year  if pd.notna(val) else _today.year
        feat[f"month_{prefix}"] = val.month if pd.notna(val) else _today.month
        feat[f"day_{prefix}"]   = val.day   if pd.notna(val) else _today.day

    X_pred = pd.DataFrame([feat])[feature_cols]
    return int(model.predict(X_pred)[0]), model.predict_proba(X_pred)[0]

# ─────────────────────────────────────────────
# ESTIMATE NEXT SERVICE DATE
# ─────────────────────────────────────────────
def estimate_next_service(row, predicted_type):
    today = datetime.date.today()
    urgency_delta = {
        "Corrective Maintenance": 7,
        "Tyre Maintenance":       30,
        "No Maintenance Record":  14,
    }
    if predicted_type in urgency_delta:
        candidate = today + datetime.timedelta(days=urgency_delta[predicted_type])
    else:
        current_odo = float(pd.to_numeric(row.get("CurrentOdometer", 0), errors="coerce") or 0)
        next_svc_km = float(pd.to_numeric(row.get("NextServiceKm",   0), errors="coerce") or 0)
        jarak       = float(pd.to_numeric(row.get("Jarak_Operasi",   0), errors="coerce") or 0)
        km_remaining = next_svc_km - current_odo
        daily_km     = jarak / 365.0 if jarak > 0 else 0
        days_left    = max(7, min(365, int(km_remaining / daily_km))) if km_remaining > 0 and daily_km > 0 else 90
        candidate    = today + datetime.timedelta(days=days_left)
    if candidate <= today:
        candidate = today + datetime.timedelta(days=urgency_delta.get(predicted_type, 90))
    return candidate

# ─────────────────────────────────────────────
# BREAKDOWN RISK GAUGE
# ─────────────────────────────────────────────
def breakdown_risk_gauge(breakdown_pct: float) -> str:
    pct = round(breakdown_pct, 1)
    if pct >= 70:
        colour, risk_label, desc = "#e63946", "High Risk", "Immediate inspection strongly recommended."
    elif pct >= 40:
        colour, risk_label, desc = "#f4a261", "Moderate Risk", "Schedule maintenance soon to prevent escalation."
    else:
        colour, risk_label, desc = "#2a9d8f", "Low Risk", "Vehicle appears in stable condition."
    return f"""
    <div class="gauge-wrapper">
        <div class="gauge-label">⚡ Breakdown Risk Before Next Service</div>
        <div class="gauge-track">
            <div class="gauge-fill" style="width:{pct}%; background:{colour};"></div>
        </div>
        <div class="gauge-pct" style="color:{colour};">{pct}%&nbsp;<span style="font-size:1rem;font-weight:600;">{risk_label}</span></div>
        <div class="gauge-desc">{desc}</div>
    </div>
    """

# ─────────────────────────────────────────────
# FLEET TABLE RENDERER
# ─────────────────────────────────────────────
def render_fleet_table(df: pd.DataFrame, key_prefix: str = ""):
    """Displays a styled dataframe + selectbox/button to open a vehicle detail page."""
    today = datetime.date.today()
    display_rows = []
    for _, row in df.iterrows():
        pc   = row["pred_class"]
        ns   = row["next_service"]
        icon = LABEL_MAP[pc][2]
        label = LABEL_MAP[pc][0]
        def _safe_int(val):
            """Convert a potentially NaN/empty value to int safely."""
            result = pd.to_numeric(val, errors="coerce")
            return int(result) if pd.notna(result) else 0

        pct = row["breakdown_pct"]
        if pct >= 80:
            risk_icon, risk_label = "🔴", "Critical"
        elif pct >= 60:
            risk_icon, risk_label = "🟠", "High"
        elif pct >= 40:
            risk_icon, risk_label = "🟡", "Moderate"
        else:
            risk_icon, risk_label = "🟢", "Low"

        display_rows.append({
            "Equipment ID":       row["EquipmentID"],
            "Category":           row.get("vehicle_category", "—"),
            "Predicted Type":     f"{icon} {label}",
            "Breakdown Risk":     f"{risk_icon} {risk_label} — {pct}%",
            "Next Service":       ns.strftime("%d %b %Y"),
            "Days Until Service": (ns - today).days,
            "Odometer (km)":      _safe_int(row.get("CurrentOdometer", 0)),
            "Breakdowns 90d":     _safe_int(row.get("breakdown_count_90d", 0)),
        })

    disp_df = pd.DataFrame(display_rows)
    st.dataframe(
        disp_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Equipment ID":       st.column_config.TextColumn("Vehicle Register Number", width="medium"),
            "Predicted Type":     st.column_config.TextColumn("Predicted Type", width="large"),
            "Breakdown Risk":     st.column_config.TextColumn("Breakdown Risk", width="large"),
            "Days Until Service": st.column_config.NumberColumn("Days Until Svc", format="%d days"),
            "Odometer (km)":      st.column_config.NumberColumn("Odometer (km)", format="%d km"),
            "Breakdowns 90d":     st.column_config.NumberColumn("BD 90d", width="small"),
        },
    )

    ids_list = disp_df["Equipment ID"].tolist()
    st.markdown(
        "<div style='font-size:0.82rem;font-weight:600;color:#6c757d;"
        "text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.3rem;'>"
        "🔎 Type or scroll to select an Vehicle Register Number, then click Open Detail</div>",
        unsafe_allow_html=True,
    )
    sel_col, btn_col = st.columns([5, 1])
    with sel_col:
        chosen = st.selectbox(
            "select",
            options=[None] + ids_list,
            index=0,
            placeholder="type to search or scroll — e.g. VKT1234…",
            format_func=lambda x: "— type or scroll to select —" if x is None else x,
            key=f"open_sel_{key_prefix}",
            label_visibility="collapsed",
        )
    with btn_col:
        if st.button("Open Detail →", key=f"open_btn_{key_prefix}", use_container_width=True):
            if chosen:
                st.session_state.selected_vehicle = chosen
                st.session_state.page = "detail"
                st.rerun()
            else:
                st.warning("Please select an Vehicle Register Number first.")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Info")
    st.info(
        "**Model Selection:** XGBoost\n\n"
        "**Maintenance Type:**\n"
        "- 🔴 Corrective (Urgent)\n"
        "- 🟡 Tyre Maintenance\n"
        "- 🟢 Preventive\n"
        "- ⚪ No Maintenance Record"
    )
    st.markdown("---")
    st.markdown("### 📅 Date")
    st.write(datetime.date.today().strftime("%d %B %Y"))
    if st.session_state.page == "detail":
        st.markdown("---")
        if st.button("⬅️ Back to Fleet Overview"):
            st.session_state.page = "fleet"
            st.session_state.selected_vehicle = None
            st.rerun()

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
with st.spinner("Loading model and training data…"):
    try:
        model, le_vehicle, le_maint, le_equip, feature_cols, train_data, equipment_ids = load_and_train()
        model_ready = True
    except Exception as e:
        model_ready = False
        equipment_ids = []
        st.error(f"⚠️ Could not connect to database: {e}")

# ══════════════════════════════════════════════════════════════════════
#  PAGE: FLEET OVERVIEW
# ══════════════════════════════════════════════════════════════════════
if st.session_state.page == "fleet":

    st.markdown('<div class="main-title">🚛 Predictive Maintenance System</div>', unsafe_allow_html=True)

    if not model_ready:
        st.error("Model not ready. Check database connection.")
        st.stop()

    with st.spinner("Loading fleet data and running predictions…"):
        try:
            fleet_df = fetch_all_vehicles()
        except Exception as e:
            st.error(f"Failed to load fleet data: {e}")
            st.stop()

    if fleet_df.empty:
        st.warning("No vehicle records found in the database.")
        st.stop()

    # Batch predictions
    pred_classes, breakdown_pcts, next_services = [], [], []
    for _, row in fleet_df.iterrows():
        try:
            pc, pp = predict_row(row, model, le_vehicle, le_maint, le_equip, feature_cols)
            ns = estimate_next_service(row, LABEL_MAP[pc][0])
        except Exception:
            pc, pp, ns = 3, [0.0, 0.0, 0.0, 1.0], datetime.date.today() + datetime.timedelta(days=14)
        pred_classes.append(pc)
        breakdown_pcts.append(round(float(pp[2]) * 100, 1))
        next_services.append(ns)

    fleet_df = fleet_df.copy()
    fleet_df["pred_class"]    = pred_classes
    fleet_df["breakdown_pct"] = breakdown_pcts
    fleet_df["next_service"]  = next_services
    fleet_df["maint_label"]   = [LABEL_MAP[c][0] for c in pred_classes]

    # ── Summary cards ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>📊 Fleet Summary</div>", unsafe_allow_html=True)
    total    = len(fleet_df)
    n_urgent = (fleet_df["pred_class"] == 2).sum()
    n_tyre   = (fleet_df["pred_class"] == 1).sum()
    n_prev   = (fleet_df["pred_class"] == 0).sum()
    n_none   = (fleet_df["pred_class"] == 3).sum()

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    cards = [
        (sc1, "all",    "🚛", total,    "Total Vehicles"),
        (sc2, "urgent", "🔴", n_urgent, "Corrective (Urgent)"),
        (sc3, "tyre",   "🟡", n_tyre,   "Tyre Maintenance"),
        (sc4, "prev",   "🟢", n_prev,   "Preventive"),
        (sc5, "none",   "⚪", n_none,   "No Record"),
    ]
    for col, cls, icon, num, lbl in cards:
        with col:
            st.markdown(f"""
            <div class="fleet-summary-card {cls}">
                <div class="fleet-card-icon">{icon}</div>
                <div class="fleet-card-num {cls}">{num}</div>
                <div class="fleet-card-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # ── Fleet notes ────────────────────────────────────────────────
    st.markdown(f"""
    <div class="notes-box">
        <strong>📝 Fleet Notes — {datetime.date.today().strftime('%d %B %Y')}</strong>
        <ul>
            <li>Predictions are generated by the XGBoost model trained on historical VMS data up to today.</li>
            <li><strong>{n_urgent} vehicle(s)</strong> require <strong>urgent corrective action</strong> — schedule workshop inspection within 7 days.</li>
            <li><strong>{n_tyre} vehicle(s)</strong> are flagged for <strong>tyre inspection or replacement</strong> — action within 30 days.</li>
            <li><strong>{n_none} vehicle(s)</strong> have <strong>no maintenance history</strong> on record — a full baseline inspection is strongly recommended.</li>
            <li>Use the <strong>filter dropdown</strong> below to narrow the list by maintenance type, or search by Equipment ID.</li>
            <li>Select any Vehicle Register Number from the table and click <strong>Open Detail →</strong> to view the full prediction report.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # ── Filter & Search ────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔍 Filter & Search Fleet</div>", unsafe_allow_html=True)
    fcol1, fcol2 = st.columns([2, 3])
    with fcol1:
        filter_type = st.selectbox(
            "Filter by Maintenance Type",
            options=MAINT_FILTER_OPTIONS,
            index=0,
            key="fleet_filter",
        )
    with fcol2:
        search_text = st.text_input(
            "Search Vehicle Register Number",
            placeholder="Type to search e.g. VKT 1234…",
            key="fleet_search",
        )

    # ── Filtered results section (shown only when filter/search active) ──
    filtered_df = fleet_df.copy()
    if filter_type != "All Vehicles":
        target_class = MAINT_FILTER_MAP.get(filter_type)
        if target_class is not None:
            filtered_df = filtered_df[filtered_df["pred_class"] == target_class]
    if search_text.strip():
        filtered_df = filtered_df[
            filtered_df["EquipmentID"].str.upper().str.contains(search_text.strip().upper(), na=False)
        ]

    if filter_type != "All Vehicles" or search_text.strip():
        st.markdown(
            f"<div class='section-header'>📋 Filtered Results "
            f"<span style='font-size:0.85rem;font-weight:400;color:#6c757d;'>"
            f"— {len(filtered_df)} vehicle(s) matched</span></div>",
            unsafe_allow_html=True,
        )
        if filtered_df.empty:
            st.info("No vehicles match the selected filter / search term.")
        else:
            render_fleet_table(filtered_df, key_prefix="filtered")
        st.markdown("---")

    # ── Full fleet table ────────────────────────────────────────────
    st.markdown(
        f"<div class='section-header'>🚛 All Vehicles "
        f"<span style='font-size:0.85rem;font-weight:400;color:#6c757d;'>"
        f"— {total} total</span></div>",
        unsafe_allow_html=True,
    )
    render_fleet_table(fleet_df, key_prefix="all")


# ══════════════════════════════════════════════════════════════════════
#  PAGE: VEHICLE DETAIL
# ══════════════════════════════════════════════════════════════════════
elif st.session_state.page == "detail":

    equipment_id = st.session_state.selected_vehicle or ""

    # Back button — top
    bcol, _ = st.columns([1, 5])
    with bcol:
        if st.button("⬅️  Back to Fleet Overview", key="back_top"):
            st.session_state.page = "fleet"
            st.session_state.selected_vehicle = None
            st.rerun()

    st.markdown('<div class="main-title">🚛 Vehicle Detail — Predictive Maintenance</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">Vehicle Register Number: <strong>{equipment_id}</strong></div>', unsafe_allow_html=True)

    if not model_ready:
        st.error("Model not ready. Check database connection.")
        st.stop()

    with st.spinner(f"Fetching data for **{equipment_id}**…"):
        try:
            vehicle_df = fetch_vehicle(equipment_id)
        except Exception as e:
            vehicle_df = pd.DataFrame()
            st.error(f"Database query failed: {e}")

    if vehicle_df.empty:
        st.warning(f"⚠️ No records found for Vehicle Register Number **{equipment_id}**.")
        st.stop()

    row = vehicle_df.iloc[0]
    pred_class, pred_proba = predict_row(row, model, le_vehicle, le_maint, le_equip, feature_cols)
    breakdown_pct          = round(float(pred_proba[2]) * 100, 1)
    maint_label, maint_key, maint_icon = LABEL_MAP[pred_class]
    priority_label   = PRIORITY_MAP.get(pred_class, "Medium")
    next_service_est = estimate_next_service(row, maint_label)

    days_until_service = (next_service_est - datetime.date.today()).days
    km_gap = max(0, int(
        (pd.to_numeric(row.get("NextServiceKm", 0), errors="coerce") or 0) -
        (pd.to_numeric(row.get("CurrentOdometer", 0), errors="coerce") or 0)
    ))
    bd30 = int(pd.to_numeric(row.get("breakdown_count_30d", 0), errors="coerce") or 0)
    bd90 = int(pd.to_numeric(row.get("breakdown_count_90d", 0), errors="coerce") or 0)

    # ── Banner ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"<div class='section-header'>📋 Vehicle Details — {row['EquipmentID']}</div>",
        unsafe_allow_html=True,
    )
    if pred_class == 2:
        banner_class, banner_text = "alert-urgent", "⚠️ URGENT ATTENTION REQUIRED"
    elif pred_class == 3:
        banner_class, banner_text = "alert-ok", "🔍 INSPECTION REQUIRED — No Prior Record"
    else:
        banner_class, banner_text = "alert-ok", "✅ No Immediate Action Needed"
    st.markdown(
        f"<div class='{banner_class}'>{maint_icon} {banner_text} — Predicted: <strong>{maint_label}</strong></div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ── Contextual analyst notes ───────────────────────────────────
    if pred_class == 2:
        notes_items = [
            f"This vehicle recorded <strong>{bd30} breakdown(s)</strong> in the last 30 days and <strong>{bd90}</strong> in the last 90 days — significantly above the safe threshold.",
            f"Corrective maintenance is predicted as <strong>imminent</strong>. Do not delay workshop inspection beyond <strong>{next_service_est.strftime('%d %b %Y')}</strong>.",
            f"Breakdown risk is <strong>{breakdown_pct}%</strong> — classified as <strong>High Risk</strong>. Consider grounding the vehicle if risk exceeds 85%.",
            f"Only <strong>{km_gap:,} km</strong> remain until the next scheduled service threshold. Continued operation without inspection may cause further damage.",
            "Recommended action: raise a <strong>priority work order</strong> immediately and notify the fleet supervisor.",
        ]
    elif pred_class == 1:
        notes_items = [
            f"Tyre wear pattern detected based on the tyre odometer reading (<strong>{int(pd.to_numeric(row.get('odometer_tyre',0),errors='coerce') or 0):,} km</strong>).",
            f"Plan tyre inspection or full replacement by <strong>{next_service_est.strftime('%d %b %Y')}</strong> — within the next 30 days.",
            f"Current odometer: <strong>{int(pd.to_numeric(row.get('CurrentOdometer',0),errors='coerce') or 0):,} km</strong> | Service target: <strong>{int(pd.to_numeric(row.get('NextServiceKm',0),errors='coerce') or 0):,} km</strong> (<strong>{km_gap:,} km</strong> remaining).",
            f"Breakdown risk: <strong>{breakdown_pct}%</strong>. Delaying tyre service increases the risk of a tyre failure on route.",
            "Recommended action: schedule a tyre inspection at the next available workshop slot.",
        ]
    elif pred_class == 3:
        notes_items = [
            "No maintenance history found for this vehicle in the VMS database.",
            f"Without a service baseline, failure risk cannot be accurately modelled. A full inspection is recommended by <strong>{next_service_est.strftime('%d %b %Y')}</strong>.",
            "After inspection, ensure all service records are entered into VMS to improve future prediction accuracy for this vehicle.",
            f"Breakdown risk is shown as <strong>{breakdown_pct}%</strong> — this figure may be <strong>understated</strong> due to missing historical data.",
            "Recommended action: conduct a comprehensive vehicle health check and register all findings in VMS.",
        ]
    else:
        notes_items = [
            f"Vehicle is currently predicted for routine <strong>Preventive Maintenance</strong> — no urgent faults detected.",
            f"Next scheduled service estimated at <strong>{next_service_est.strftime('%d %b %Y')}</strong> ({days_until_service} days from today).",
            f"Remaining distance until next service: <strong>{km_gap:,} km</strong> — current odometer at <strong>{int(pd.to_numeric(row.get('CurrentOdometer',0),errors='coerce') or 0):,} km</strong>.",
            f"Breakdown risk: <strong>{breakdown_pct}%</strong> — Low. Vehicle is operating within normal parameters.",
            "Recommended action: maintain the current scheduled service plan and monitor odometer readings weekly.",
        ]

    notes_html = "".join(f"<li>{item}</li>" for item in notes_items)
    st.markdown(f"""
    <div class="notes-box">
        <strong>📝 Analyst Notes</strong>
        <ul>{notes_html}</ul>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='metric-card {maint_key}'>"
            f"<div class='card-label'>Predicted Maintenance Type</div>"
            f"<div class='card-value'>{maint_icon} {maint_label}</div>"
            f"</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='card-label'>Priority Level</div>"
            f"<div class='card-value'>{priority_label}</div>"
            f"</div>", unsafe_allow_html=True)
    with c3:
        last_date = pd.to_datetime(row.get("last_service_date", ""), errors="coerce")
        last_str  = last_date.strftime("%d %b %Y") if pd.notna(last_date) else "N/A"
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='card-label'>Last Service Date</div>"
            f"<div class='card-value'>{last_str}</div>"
            f"</div>", unsafe_allow_html=True)
    with c4:
        st.markdown(
            f"<div class='metric-card urgent' style='border-left-color:#4361ee;background:#f0f4ff'>"
            f"<div class='card-label'>Estimated Next Service</div>"
            f"<div class='card-value'>{next_service_est.strftime('%d %b %Y')}</div>"
            f"</div>", unsafe_allow_html=True)

    st.markdown("")

    # ── Breakdown risk gauge ───────────────────────────────────────
    st.markdown(breakdown_risk_gauge(breakdown_pct), unsafe_allow_html=True)

    # ── Vehicle info / stats columns ──────────────────────────────
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("<div class='section-header'>🚛 Vehicle Information</div>", unsafe_allow_html=True)
        for k, v in {
            "Vehicle Register Number":      row.get("EquipmentID", "—"),
            "Vehicle Category":  row.get("vehicle_category", "—"),
            "Current Odometer":  f"{int(pd.to_numeric(row.get('CurrentOdometer', 0), errors='coerce') or 0):,} km",
            "Next Service At":   f"{int(pd.to_numeric(row.get('NextServiceKm', 0), errors='coerce') or 0):,} km",
            "Km Until Service":  f"{km_gap:,} km",
            "Distance Operated": f"{int(pd.to_numeric(row.get('Jarak_Operasi', 0), errors='coerce') or 0):,} km",
            "Total Services":    int(pd.to_numeric(row.get("service_count", 0), errors="coerce") or 0),
        }.items():
            st.markdown(f"**{k}:** {v}")

    with col_right:
        st.markdown("<div class='section-header'>🔧 Breakdown & Odometer Stats</div>", unsafe_allow_html=True)
        odo_last = pd.to_datetime(row.get("OdometerLastUpdate", ""), errors="coerce")
        for k, v in {
            "Breakdowns (Last 30 days)": bd30,
            "Breakdowns (Last 90 days)": bd90,
            "Odometer — Maintenance": f"{int(pd.to_numeric(row.get('odometer_maintenance', 0), errors='coerce') if pd.notna(pd.to_numeric(row.get('odometer_maintenance', 0), errors='coerce')) else 0):,} km",
            "Odometer — Repair":         f"{int(pd.to_numeric(row.get('odometer_repair', 0), errors='coerce') or 0):,} km",
            "Odometer — Tyre": f"{int(pd.to_numeric(row.get('odometer_tyre', 0), errors='coerce') if pd.notna(pd.to_numeric(row.get('odometer_tyre', 0), errors='coerce')) else 0):,} km",
            "Last Odometer Update":      str(odo_last.date()) if pd.notna(odo_last) else "N/A",
        }.items():
            st.markdown(f"**{k}:** {v}")

    # ── Breakdown trend charts ────────────────────────────────────
    st.markdown("<div class='section-header'>📈 Breakdown Trend Analysis</div>", unsafe_allow_html=True)

    # Derive monthly-like trend from available window data:
    # We have bd30 (last 30d) and bd90 (last 90d).
    # From these we can infer:
    #   - last 30 days  = bd30
    #   - 31–60 days ago = estimated from (bd90 - bd30) split evenly over 2 months
    #   - 61–90 days ago = remaining share
    # Odometer readings add a maintenance-activity timeline.
    today_dt = datetime.date.today()

    # Build 3-month rolling breakdown estimate
    bd_mid  = max(0, bd90 - bd30)          # total in days 31-90
    bd_m2   = round(bd_mid * 0.55)         # ~55% weight to the closer month
    bd_m3   = max(0, bd_mid - bd_m2)       # remaining for oldest month

    month_labels = [
        (today_dt - datetime.timedelta(days=75)).strftime("%b %Y"),
        (today_dt - datetime.timedelta(days=45)).strftime("%b %Y"),
        today_dt.strftime("%b %Y"),
    ]
    breakdown_trend = pd.DataFrame({
        "Month":      month_labels,
        "Breakdowns": [int(bd_m3), int(bd_m2), int(bd30)],
    })

    # Odometer activity by service type
    odo_maint = int(pd.to_numeric(row.get("odometer_maintenance"), errors="coerce") or 0) if pd.notna(pd.to_numeric(row.get("odometer_maintenance"), errors="coerce")) else 0
    odo_repair = int(pd.to_numeric(row.get("odometer_repair", 0), errors="coerce") or 0) if pd.notna(pd.to_numeric(row.get("odometer_repair", 0), errors="coerce")) else 0
    odo_tyre   = int(pd.to_numeric(row.get("odometer_tyre", 0), errors="coerce") or 0) if pd.notna(pd.to_numeric(row.get("odometer_tyre", 0), errors="coerce")) else 0

    odo_df = pd.DataFrame({
        "Service Type": ["Maintenance", "Repair", "Tyre"],
        "Odometer at Last Service (km)": [odo_maint, odo_repair, odo_tyre],
    })

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown(
            "<div style='font-size:0.9rem;font-weight:600;color:#1a1a2e;margin-bottom:0.4rem;'>"
            "🔴 Monthly Breakdown Count (Last 3 Months)</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:0.78rem;color:#6c757d;margin-bottom:0.6rem;'>"
            "Estimated from 30-day and 90-day breakdown window data.</div>",
            unsafe_allow_html=True,
        )
        breakdown_trend["Breakdowns"] = pd.to_numeric(
            breakdown_trend["Breakdowns"],
            errors="coerce"
        ).fillna(0)

        st.line_chart(
            data=breakdown_trend,
            x="Month",
            y="Breakdowns",
            use_container_width=True,
            height=260,
        )
        # Trend indicator
        if bd30 > bd_m2:
            trend_msg = "⚠️ Breakdowns are **increasing** — deterioration trend detected."
            st.warning(trend_msg)
        elif bd30 < bd_m2:
            trend_msg = "✅ Breakdowns are **decreasing** — condition improving."
            st.success(trend_msg)
        else:
            trend_msg = "➡️ Breakdown frequency is **stable**."
            st.info(trend_msg)

    with chart_col2:
        st.markdown(
            "<div style='font-size:0.9rem;font-weight:600;color:#1a1a2e;margin-bottom:0.4rem;'>"
            "🔧 Odometer at Last Service by Type</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='font-size:0.78rem;color:#6c757d;margin-bottom:0.6rem;'>"
            "Shows how far the vehicle had travelled at each last recorded service.</div>",
            unsafe_allow_html=True,
        )
        odo_df["Odometer at Last Service (km)"] = pd.to_numeric(
            odo_df["Odometer at Last Service (km)"],
            errors="coerce"
        ).fillna(0)

        st.bar_chart(
            data=odo_df,
            x="Service Type",
            y="Odometer at Last Service (km)",
            use_container_width=True,
            height=260,
        )
        current_odo = int(pd.to_numeric(row.get("CurrentOdometer", 0), errors="coerce") or 0)
        max_odo_svc = max(odo_maint, odo_repair, odo_tyre)
        if max_odo_svc > 0:
            gap = current_odo - max_odo_svc
            if gap > 0:
                st.info(f"🛣️ Vehicle has travelled **{gap:,} km** since the last recorded service activity.")
            else:
                st.info("🛣️ Last service odometer is at or above current odometer — record may reflect recent service.")

    # ── Model confidence chart ─────────────────────────────────────
    st.markdown("<div class='section-header'>📊 Model Confidence</div>", unsafe_allow_html=True)
    proba_df = pd.DataFrame({
        "Maintenance Type": [
            "Preventive Maintenance", "Tyre Maintenance",
            "Corrective Maintenance", "No Maintenance Record",
        ],
        "Confidence (%)": [round(p * 100, 1) for p in pred_proba],
    })
    proba_df["Confidence (%)"] = pd.to_numeric(
        proba_df["Confidence (%)"],
        errors="coerce"
    ).fillna(0)

    st.bar_chart(
        data=proba_df,
        x="Maintenance Type",
        y="Confidence (%)",
        use_container_width=True,
    )

    # ── Recommendation ─────────────────────────────────────────────
    st.markdown("<div class='section-header'>💡 Recommendation</div>", unsafe_allow_html=True)
    if pred_class == 2:
        st.error(
            f"**Action Required Immediately.** This vehicle is predicted to require **Corrective Maintenance** "
            f"within the next 7 days. High breakdown frequency ({bd30} in last 30 days) and odometer proximity "
            f"to the next service threshold indicate urgent attention. "
            f"Schedule workshop inspection before **{next_service_est.strftime('%d %b %Y')}**. "
            f"Breakdown risk: **{breakdown_pct}%**."
        )
    elif pred_class == 1:
        st.warning(
            f"**Tyre Maintenance Recommended.** The vehicle's tyre odometer reading suggests tyre wear. "
            f"Plan a tyre inspection or replacement within the next 30 days. "
            f"Suggested date: **{next_service_est.strftime('%d %b %Y')}**. "
            f"Breakdown risk: **{breakdown_pct}%**."
        )
    elif pred_class == 3:
        st.info(
            f"**No Maintenance Record Found.** This vehicle has no recorded maintenance history. "
            f"Schedule a full inspection to establish a service baseline by **{next_service_est.strftime('%d %b %Y')}**. "
            f"Breakdown risk: **{breakdown_pct}%**."
        )
    else:
        st.success(
            f"**Routine Preventive Maintenance.** Vehicle is in good condition. "
            f"Schedule the next preventive service by **{next_service_est.strftime('%d %b %Y')}** "
            f"to keep it roadworthy. Breakdown risk: **{breakdown_pct}%**."
        )

    # Back button — bottom
    st.markdown("---")
    bcol2, _ = st.columns([1, 5])
    with bcol2:
        if st.button("⬅️  Back to Fleet Overview", key="back_bottom"):
            st.session_state.page = "fleet"
            st.session_state.selected_vehicle = None
            st.rerun()