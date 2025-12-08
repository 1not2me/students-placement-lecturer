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
              direction:rtl;
              text-align:center;
              margin:0;
              padding-top:120px;
              color:#111827;
            }
            .box{
              display:inline-block;
              padding:32px 40px;
              border-radius:18px;
              background:#ffffff;
              box-shadow:0 10px 30px rgba(15,23,42,.08);
              border:1px solid #e5e7eb;
            }
            h1{margin:0 0 12px;font-size:26px;}
            p{margin:0;color:#6b7280;}
          </style>
        </head>
        <body>
          <div class="box">
            <h1>⚙️ האתר סגור כרגע</h1>
            <p>הגישה למערכת השיבוץ מוגבלת זמנית.</p>
          </div>
        </body>
        </html>
        """
        return Markup(html), 503

# ========= מודל ניקוד =========
@dataclass
class Weights:
    w_field: float = 0.50   # תחום
    w_city: float = 0.05    # עיר
    w_special: float = 0.45 # בקשות מיוחדות

# עמודות סטודנטים
STU_COLS = {
    "id": ["מספר תעודת זהות", "תעודת זהות", "ת\"ז", "תז", "תעודת זהות הסטודנט"],
    "first": ["שם פרטי"],
    "last": ["שם משפחה"],
    # חדש – כתובת, למקרה שאין עמודת עיר
    "address": ["כתובת", "כתובת מלאה", "כתובת הסטודנט"],
    "city": ["עיר מגורים", "עיר"],
    "phone": ["טלפון", "מספר טלפון"],
    "email": ["דוא\"ל", "דוא״ל", "אימייל", "כתובת אימייל", "כתובת מייל"],
    "preferred_field": ["תחום מועדף", "תחומים מועדפים"],
    "special_req": ["בקשה מיוחדת"],
    "partner": ["בן/בת זוג להכשרה", "בן\\בת זוג להכשרה", "בן/בת זוג", "בן\\בת זוג"]
}

# עמודות אתרים
SITE_COLS = {
    "name": ["מוסד / שירות הכשרה", "מוסד", "שם מוסד ההתמחות", "שם המוסד", "מוסד ההכשרה"],
    "field": ["תחום ההתמחות", "תחום התמחות"],
    "street": ["רחוב"],
    "city": ["עיר"],
    "capacity": ["מספר סטודנטים שניתן לקלוט השנה", "מספר סטודנטים שניתן לקלוט", "קיבולת"],
    "sup_first": ["שם פרטי"],
    "sup_last": ["שם משפחה"],
    "phone": ["טלפון"],
    "email": ["אימייל", "כתובת מייל", "דוא\"ל", "דוא״ל"],
    "review": ["חוות דעת מדריך"]
}

# ========= פונקציות עזר =========
def pick_col(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    for opt in options:
        if opt in df.columns:
            return opt
    return None

def read_any(uploaded) -> pd.DataFrame:
    name = (uploaded.filename or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded, encoding="utf-8-sig")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    return pd.read_csv(uploaded, encoding="utf-8-sig")

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

    # עיר – קודם מנסים עמודת "עיר", ואם אין – מחלצים מהכתובת (החלק אחרי הפסיק)
    city_col = pick_col(out, STU_COLS["city"])
    if city_col:
        out["stu_city"] = out[city_col]
    else:
        addr_col = pick_col(out, STU_COLS.get("address", []))
        if addr_col:
            out["stu_city"] = out[addr_col].apply(
                lambda x: str(x).split(",")[-1].strip() if isinstance(x, str) and "," in x else ""
            )
        else:
            out["stu_city"] = ""

    pref_col = pick_col(out, STU_COLS["preferred_field"])
    out["stu_pref"] = out[pref_col] if pref_col else ""

    req_col = pick_col(out, STU_COLS["special_req"])
    out["stu_req"] = out[req_col] if req_col else ""

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
    out["capacity_left"] = out["site_capacity"].astype(int)

    sup_first = pick_col(out, SITE_COLS["sup_first"])
    sup_last = pick_col(out, SITE_COLS["sup_last"])
    out["שם המדריך"] = ""
    if sup_first or sup_last:
        ff = out[sup_first] if sup_first else ""
        ll = out[sup_last] if sup_last else ""
        out["שם המדריך"] = (ff.astype(str) + " " + ll.astype(str)).str.strip()

    for c in ["site_name", "site_field", "site_city", "שם המדריך"]:
        out[c] = out[c].apply(normalize_text)
    return out

# --- ציון + פירוק לפי 50/45/5 ---
def compute_score_with_explain(stu: pd.Series, site: pd.Series, W: Weights):
    stu_city   = normalize_text(stu.get("stu_city", "")).lower()
    site_city  = normalize_text(site.get("site_city", "")).lower()
    stu_pref   = normalize_text(stu.get("stu_pref", "")).lower()
    site_field = normalize_text(site.get("site_field", "")).lower()
    stu_req    = normalize_text(stu.get("stu_req", ""))

    # 1) תחום – 50%
    # לסטודנט/ית יכולים להיות כמה תחומים: "רווחה; שיקום; קהילה"
    if stu_pref:
        tokens = [t.strip() for t in stu_pref.replace(";", ",").split(",") if t.strip()]
        if tokens:
            field_component = 100 if any(tok in site_field for tok in tokens) else 0
        else:
            field_component = 70
    else:
        field_component = 70  # ניטרלי חלקי כשאין העדפה בכלל

    # 2) עיר – 5%
    if stu_city and site_city:
        city_component = 100 if stu_city == site_city else 0
    else:
        city_component = 50  # ניטרלי כשאין מידע מלא

    # 3) בקשות מיוחדות – 45%
    stu_req_lower = stu_req.lower()
    # "קרוב לבית" – אם יש גם עיר, בודקים התאמה; אם אין – ניטרלי
    if "קרוב" in stu_req_lower:
        if stu_city and site_city:
            special_component = 100 if stu_city == site_city else 0
        else:
            special_component = 50  # לא מענישים על חוסר מידע
    # "אזור צפון" – התאמה חלקית אם המוסד בצפון
    elif "צפון" in stu_req_lower:
        north_cities = ["צפת", "כרמיאל", "נהריה", "עכו", "קריית שמונה", "טבריה", "חורפיש"]
        north_cities = [c.lower() for c in north_cities]
        if site_city in north_cities:
            special_component = 75  # חלקי – בסביבות 80–85% יחד עם תחום
        else:
            special_component = 50
    else:
        # אין בקשה מיוחדת – ניטרלי
        special_component = 50

    parts = {
        "התאמת תחום": round(W.w_field * field_component),
        "מרחק/גיאוגרפיה": round(W.w_city * city_component),
        "בקשות מיוחדות": round(W.w_special * special_component),
        "עדיפויות הסטודנט/ית": 0
    }
    score = int(np.clip(sum(parts.values()), 0, 100))
    return score, parts

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

        # אם אין אתר שעובר את סינון המדריכים – בוחרים את הטוב מכל הפנויים
        if filtered.empty:
            all_sites = sites_df[sites_df["capacity_left"] > 0].copy()
            if all_sites.empty:
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
            all_sites[["score", "_parts"]] = all_sites.apply(score_row, axis=1)
            filtered = all_sites.sort_values("score", ascending=False).head(1)
        else:
            filtered = filtered.sort_values("score", ascending=False)

        chosen = filtered.iloc[0]
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
            "עיר המוסד": chosen.get("site_city", ""),
            "תחום ההתמחות במוסד": chosen["site_field"],
            "שם המדריך": sup_name,
            "אחוז התאמה": int(chosen["score"]),
            "_expl": chosen["_parts"]
        })

    return pd.DataFrame(results)

# --- יצירת XLSX ---
def df_to_xlsx_bytes(df: pd.DataFrame, sheet_name: str = "שיבוץ") -> bytes:
    xlsx_io = BytesIO()
    import xlsxwriter

    with pd.ExcelWriter(xlsx_io, engine="xlsxwriter") as writer:
        cols = list(df.columns)
        has_match_col = "אחוז התאמה" in cols
        if has_match_col:
            cols = [c for c in cols if c != "אחוז התאמה"] + ["אחוז התאמה"]

        df[cols].to_excel(writer, index=False, sheet_name=sheet_name)

        if has_match_col:
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            red_fmt = workbook.add_format({"font_color": "red"})
            col_idx = len(cols) - 1
            worksheet.set_column(col_idx, col_idx, 12, red_fmt)

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
        "explanations": None,  # <- לרינדור הדפדוף
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
            last_results_df = base_df.copy()

            # טבלת תוצאות להצגה
            df_show = pd.DataFrame({
                "אחוז התאמה": base_df["אחוז התאמה"].astype(int),
                "שם הסטודנט/ית": (base_df["שם פרטי"].astype(str) + " " + base_df["שם משפחה"].astype(str)).str.strip(),
                "תעודת זהות": base_df["ת\"ז הסטודנט"],
                "תחום התמחות": base_df["תחום ההתמחות במוסד"],
                "עיר המוסד": base_df["עיר המוסד"],
                "שם מקום ההתמחות": base_df["שם מקום ההתמחות"],
                "שם המדריך/ה": base_df["שם המדריך"],
            }).sort_values("אחוז התאמה", ascending=False)

            # סיכום לפי אתר/תחום/מדריך
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

            # קיבולות מול שיבוץ בפועל
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
                })
            cap_df = pd.DataFrame(cap_rows).sort_values("שם מקום ההתמחות")

            # הסברים לכל הסטודנטים
            explanations = []
            for _, r in base_df.iterrows():
                explanations.append({
                    "student": f"{r['שם פרטי']} {r['שם משפחה']}",
                    "site": r["שם מקום ההתמחות"],
                    "score": int(r["אחוז התאמה"]),
                    "parts": r["_expl"]  # {"התאמת תחום":..., "מרחק/גיאוגרפיה":..., ...}
                })

            context.update({
                "results": df_show.to_dict(orient="records"),
                "summary": summary_df.to_dict(orient="records"),
                "capacities": cap_df.to_dict(orient="records"),
                "explanations": explanations,
                "error": None
            })

        except Exception as e:
            context["error"] = f"שגיאה במהלך השיבוץ: {e}"

    return render_template("index.html", **context)

# ========= הורדות =========
@app.route("/download/results")
def download_results():
    global last_results_df
    if last_results_df is None or last_results_df.empty:
        return "אין נתוני שיבוץ להורדה", 400

    df_show = pd.DataFrame({
        "אחוז התאמה": last_results_df["אחוז התאמה"].astype(int),
        "שם הסטודנט/ית": (last_results_df["שם פרטי"].astype(str) + " " + last_results_df["שם משפחה"].astype(str)).str.strip(),
        "תעודת זהות": last_results_df["ת\"ז הסטודנט"],
        "תחום התמחות": last_results_df["תחום ההתמחות במוסד"],
        "עיר המוסד": last_results_df["עיר המוסד"],
        "שם מקום ההתמחות": last_results_df["שם מקום ההתמחות"],
        "שם המדריך/ה": last_results_df["שם המדריך"],
    }).sort_values("אחוז התאמה", ascending=False)

    data = df_to_xlsx_bytes(df_show, sheet_name="תוצאות")
    return send_file(
        BytesIO(data),
        as_attachment=True,
        download_name="student_site_matching.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@app.route("/download/summary")
def download_summary():
    global last_summary_df
    if last_summary_df is None or last_summary_df.empty:
        return "אין טבלת סיכום להורדה", 400

    data = df_to_xlsx_bytes(last_summary_df, sheet_name="סיכום")
    return send_file(
        BytesIO(data),
        as_attachment=True,
        download_name="student_site_summary.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
