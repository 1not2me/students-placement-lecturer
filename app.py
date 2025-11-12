# -*- coding: utf-8 -*-
import os
from flask import Flask, render_template, request, send_file
from markupsafe import Markup
import pandas as pd
import numpy as np
from io import BytesIO
from dataclasses import dataclass
from typing import Any, List, Optional

app = Flask(__name__)

# ---------- מצב תחזוקה / סגור ----------
@app.before_request
def maintenance_mode():
    """
    אם במשתני סביבה יש MAINTENANCE_MODE=1
    כל בקשה תחזיר דף 'האתר סגור'.
    לפתיחה: לשנות ל-0 או להסיר את המשתנה.
    """
    if os.getenv("MAINTENANCE_MODE", "0") == "1":
        html = """
        <html lang="he" dir="rtl">
        <head>
          <meta charset="utf-8">
          <title>האתר סגור</title>
          <style>
            body{
              font-family:system-ui,-apple-system,Segoe UI,Heebo,Arial;
              background:#f8fafc;
              display:flex;
              align-items:center;
              justify-content:center;
              height:100vh;
              margin:0;
              color:#0f172a;
            }
            .box{
              padding:2rem 2.4rem;
              border-radius:18px;
              background:#ffffff;
              box-shadow:0 18px 45px rgba(15,23,42,0.12);
              border:1px solid rgba(148,163,253,0.3);
              max-width:520px;
              text-align:center;
            }
            h1{
              margin:0 0 0.75rem;
              font-size:1.8rem;
            }
            p{
              margin:0;
              font-size:1rem;
              color:#4b5563;
            }
          </style>
        </head>
        <body>
          <div class="box">
            <h1>האתר כעת סגור לתחזוקה 🛠️</h1>
            <p>אנחנו מבצעים עדכונים במערכת. אנא נסו שוב מאוחר יותר.</p>
          </div>
        </body>
        </html>
        """
        return Markup(html)

# ---------- קבועי עמודות אפשריים ----------

STU_COLS = {
    "id": ["תז", "ת\"ז", "מספר זהות", "id", "ID", "stu_id"],
    "first": ["שם פרטי", "פרטי", "first_name", "first", "stu_first"],
    "last": ["שם משפחה", "משפחה", "last_name", "last", "stu_last"],
    "city": ["עיר מגורים", "עיר הסטודנט", "city", "stu_city"],
    "preferred_field": ["תחום מועדף", "תחום התמחות מועדף", "שדה מועדף", "pref_field", "preferred_field"],
    "special_req": ["בקשות מיוחדות סטודנט", "בקשות מיוחדות", "special_req", "דרישות מיוחדות"],
}

SITE_COLS = {
    "name": ["שם מקום ההתמחות", "שם מוסד", "מוסד", "site_name"],
    "field": ["תחום התמחות", "תחום ההתמחות במוסד", "field", "site_field"],
    "city": ["עיר המוסד", "עיר", "site_city"],
    "capacity": ["קיבולת", "מספר סטודנטים שניתן לקלוט (1 או 2)", "capacity", "site_capacity"],
    "special_req": ["בקשות מיוחדות ממוסד", "בקשות מיוחדות", "site_special_req"],
    "supervisor": ["שם המדריך", "שם מדריך", "supervisor_name"],
}

# ---------- משקולות ברירת מחדל ----------

@dataclass
class Weights:
    field: float = 0.5
    geo: float = 0.25
    special: float = 0.15
    pref: float = 0.10

# ========= פונקציות עזר =========

def read_any(uploaded) -> pd.DataFrame:
    filename = uploaded.filename.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)

def pick_col(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for opt in options:
        for c_low, c_real in cols_lower.items():
            if opt.lower() == c_low:
                return c_real
    for opt in options:
        for c_low, c_real in cols_lower.items():
            if opt.lower() in c_low:
                return c_real
    return None

def normalize_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()

# --- סטודנטים ---
def resolve_students(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["stu_id"] = out[pick_col(out, STU_COLS["id"])]
    out["stu_first"] = out[pick_col(out, STU_COLS["first"])]
    out["stu_last"] = out[pick_col(out, STU_COLS["last"])]
    out["stu_city"] = out[pick_col(out, STU_COLS["city"])] if pick_col(out, STU_COLS["city"]) else ""
    out["stu_pref"] = out[pick_col(out, STU_COLS["preferred_field"])] if pick_col(out, STU_COLS["preferred_field"]) else ""
    out["stu_req"] = out[pick_col(out, STU_COLS["special_req"])] if pick_col(out, STU_COLS["special_req"]) else ""

    for c in ["stu_id", "stu_first", "stu_last", "stu_city", "stu_pref", "stu_req"]:
        out[c] = out[c].apply(normalize_text)
    return out

# --- אתרים ---
def resolve_sites(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["site_name"] = out[pick_col(out, SITE_COLS["name"])]
    out["site_field"] = out[pick_col(out, SITE_COLS["field"])]
    out["site_city"] = out[pick_col(out, SITE_COLS["city"])]

    cap_col = pick_col(out, SITE_COLS["capacity"])
    if cap_col:
        out["site_capacity"] = pd.to_numeric(out[cap_col], errors="coerce").fillna(1).astype(int)
    else:
        out["site_capacity"] = 1

    spec_col = pick_col(out, SITE_COLS["special_req"])
    out["site_req"] = out[spec_col] if spec_col else ""

    sup_col = pick_col(out, SITE_COLS["supervisor"])
    out["site_supervisor"] = out[sup_col] if sup_col else ""

    for c in ["site_name", "site_field", "site_city", "site_req", "site_supervisor"]:
        out[c] = out[c].apply(normalize_text)

    out["capacity_left"] = out["site_capacity"].copy()
    return out

# --- חישוב ציון והתפלגות ---
def compute_score_with_explain(stu, site, W: Weights):
    parts = {}

    # התאמת תחום
    if stu["stu_pref"] and site["site_field"]:
        parts["התאמת תחום"] = 100 if stu["stu_pref"] in site["site_field"] else 0
    else:
        parts["התאמת תחום"] = 50

    # מרחק / גיאוגרפיה (כאן דמוי־לוגיקה פשוטה)
    if stu["stu_city"] and site["site_city"]:
        parts["מרחק/גיאוגרפיה"] = 100 if stu["stu_city"] == site["site_city"] else 40
    else:
        parts["מרחק/גיאוגרפיה"] = 60

    # בקשות מיוחדות מוסד/סטודנט
    parts["בקשות מיוחדות"] = 100

    # עדיפויות הסטודנט/ית (כאן בגרסה פשוטה)
    parts["עדיפויות הסטודנט/ית"] = 80

    score = (
        parts["התאמת תחום"] * W.field +
        parts["מרחק/גיאוגרפיה"] * W.geo +
        parts["בקשות מיוחדות"] * W.special +
        parts["עדיפויות הסטודנט/ית"] * W.pref
    ) / 100.0 * 100

    return round(score), parts

# --- אלגוריתם שיבוץ חמדני ---
def greedy_match(students_df: pd.DataFrame, sites_df: pd.DataFrame, W: Weights) -> pd.DataFrame:
    results = []
    supervisor_count = {}  # עד 2 סטודנטים לכל מדריך (ניתן לשנות)

    for _, s in students_df.iterrows():
        cand = sites_df[sites_df["capacity_left"] > 0].copy()

        # אין בכלל מקומות פנויים
        if cand.empty:
            results.append({
                "ת\"ז הסטודנט": s["stu_id"],
                "שם פרטי": s["stu_first"],
                "שם משפחה": s["stu_last"],
                "שם מקום ההתמחות": "לא שובץ",
                "עיר המוסד": "",
                "תחום ההתמחות במוסד": "",
                "שם המדריך": "",
                "אחוז התאמה": 0,
                "_expl": {
                    "התאמת תחום": 0,
                    "מרחק/גיאוגרפיה": 0,
                    "בקשות מיוחדות": 0,
                    "עדיפויות הסטודנט/ית": 0
                }
            })
            continue

        # מחשבים ציון לכל אתר
        def score_row(r):
            sc, parts = compute_score_with_explain(s, r, W)
            return pd.Series({"score": sc, "_parts": parts})

        cand[["score", "_parts"]] = cand.apply(score_row, axis=1)

        # מסננים לפי מגבלת מדריך (עד 2 סטודנטים למשל)
        def allowed_supervisor(r):
            sup = r.get("שם המדריך", "")
            return supervisor_count.get(sup, 0) < 2

        filtered = cand[cand.apply(allowed_supervisor, axis=1)]

        # אם אין אתר לאחר סינון – לוקחים מהמקורי
        if filtered.empty:
            filtered = cand

        # בוחרים את האתר עם הציון הגבוה
        chosen = filtered.sort_values("score", ascending=False).iloc[0]
        idx = chosen.name

        # מעדכנים קיבולת
        sites_df.at[idx, "capacity_left"] -= 1

        # מעדכנים ספירת סטודנטים למדריך
        sup_name = chosen.get("שם המדריך", "")
        supervisor_count[sup_name] = supervisor_count.get(sup_name, 0) + 1

        # שורת תוצאה
        results.append({
            "ת\"ז הסטודנט": s["stu_id"],
            "שם פרטי": s["stu_first"],
            "שם משפחה": s["stu_last"],
            "שם מקום ההתמחות": chosen["site_name"],
            "עיר המוסד": chosen["site_city"],
            "תחום ההתמחות במוסד": chosen["site_field"],
            "שם המדריך": chosen.get("site_supervisor", ""),
            "אחוז התאמה": chosen["score"],
            "_expl": chosen["_parts"]
        })

    return pd.DataFrame(results)

# ========= עזר ל-XLSX =========

def df_to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    xlsx_io = BytesIO()
    with pd.ExcelWriter(xlsx_io, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        # עיצוב ראשי טבלה
        header_fmt = workbook.add_format({
            "bold": True,
            "bg_color": "#EEF2FF",
            "font_color": "#111827",
            "border": 1
        })
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_fmt)

        # התאמת רוחב עמודות
        for i, col in enumerate(df.columns):
            max_len = max([len(str(v)) for v in df[col]] + [len(col)]) + 2
            worksheet.set_column(i, i, max_len)

    xlsx_io.seek(0)
    return xlsx_io.getvalue()

# ========= משתנים גלובליים =========
last_results_df: Optional[pd.DataFrame] = None
last_summary_df: Optional[pd.DataFrame] = None

# ========= ראוט ראשי =========
@app.route("/", methods=["GET", "POST"])
def index():
    global last_results_df, last_summary_df

    context = {
        "results": None,
        "summary": None,
        "capacities": None,
        "expl_for_first": None,
        "explanations": None,
        "error": None
    }

    if request.method == "POST":
        students_file = request.files.get("students_file")
        sites_file = request.files.get("sites_file")

        if not students_file or not sites_file:
            context["error"] = "יש להעלות גם קובץ סטודנטים וגם קובץ אתרי התמחות."
            return render_template("index.html", **context)

        try:
            df_students_raw = read_any(students_file)
            df_sites_raw = read_any(sites_file)

            students = resolve_students(df_students_raw)
            sites = resolve_sites(df_sites_raw)

            base_df = greedy_match(students, sites, Weights())

            # מיון התוצאות ושמירה למשתנה גלובלי
            base_sorted = base_df.sort_values("אחוז התאמה", ascending=False).reset_index(drop=True)
            last_results_df = base_sorted.copy()

            # טבלת תוצאות לתצוגה
            df_show = pd.DataFrame({
                "אחוז התאמה": base_sorted["אחוז התאמה"].astype(int),
                "שם הסטודנט/ית": (base_sorted["שם פרטי"].astype(str) + " " + base_sorted["שם משפחה"].astype(str)).str.strip(),
                "תעודת זהות": base_sorted["ת\"ז הסטודנט"],
                "תחום התמחות": base_sorted["תחום ההתמחות במוסד"],
                "עיר המוסד": base_sorted["עיר המוסד"],
                "שם מקום ההתמחות": base_sorted["שם מקום ההתמחות"],
                "שם המדריך/ה": base_sorted["שם המדריך"],
            })

            # טבלת סיכום למוסדות
            summary_df = (
                base_df
                .groupby(["שם מקום ההתמחות", "תחום ההתמחות במוסד", "שם המדריך"])
                .agg({
                    "ת\"ז הסטודנט": "count",
                    "שם פרטי": list,
                    "שם משפחה": list
                }).reset_index()
            )
            summary_df.rename(columns={"ת\"ז הסטודנט": "כמה סטודנטים"}, inplace=True)
            summary_df["המלצת שיבוץ"] = summary_df.apply(
                lambda row: " + ".join(
                    [f"{f} {l}" for f, l in zip(row["שם פרטי"], row["שם משפחה"])]
                ),
                axis=1
            )
            summary_df = summary_df[[
                "שם מקום ההתמחות",
                "תחום ההתמחות במוסד",
                "שם המדריך",
                "כמה סטודנטים",
                "המלצת שיבוץ"
            ]]
            last_summary_df = summary_df.copy()

            # קיבולת מול שיבוץ בפועל
            caps = sites.groupby("site_name")["site_capacity"].sum().to_dict()
            assigned = base_df.groupby("שם מקום ההתמחות")["ת\"ז הסטודנט"].count().to_dict()
            cap_rows = []
            for site_name, capacity in caps.items():
                used = int(assigned.get(site_name, 0))
                cap_rows.append({
                    "שם מקום ההתמחות": site_name,
                    "קיבולת": int(capacity),
                    "שובצו בפועל": used,
                    "יתרה/חוסר": int(capacity - used)
