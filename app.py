# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
from io import BytesIO
from dataclasses import dataclass
from typing import Any, List, Optional

app = Flask(__name__)

# ========= מודל ניקוד =========
@dataclass
class Weights:
    w_field: float = 0.50
    w_city: float = 0.05
    w_special: float = 0.45

# עמודות סטודנטים
STU_COLS = {
    "id": ["מספר תעודת זהות", "תעודת זהות", "ת\"ז", "תז", "תעודת זהות הסטודנט"],
    "first": ["שם פרטי"],
    "last": ["שם משפחה"],
    "address": ["כתובת", "כתובת הסטודנט", "רחוב"],
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
    # ברירת מחדל
    return pd.read_csv(uploaded, encoding="utf-8-sig")

def normalize_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()

def compute_score_with_explain(stu: pd.Series, site: pd.Series, W: Weights):
    # נורמליזציה
    stu_city   = normalize_text(stu.get("stu_city", "")).lower()
    site_city  = normalize_text(site.get("site_city", "")).lower()
    stu_pref   = normalize_text(stu.get("stu_pref", "")).lower()
    site_field = normalize_text(site.get("site_field", "")).lower()
    stu_req    = normalize_text(stu.get("stu_req", ""))

    # ----- 1) תחום – 50% -----
    # יש תחום מועדף והוא מופיע בתחום המוסד → 100
    # יש תחום מועדף והוא לא מתאים → 0
    # אין תחום מועדף → ניטרלי 70
    if stu_pref:
        if stu_pref in site_field:
            field_component = 100
        else:
            field_component = 0
    else:
        field_component = 70

    # ----- 2) עיר – 5% -----
    # אותה עיר → 100, אחרת → 0, בלי עיר → 50 (ניטרלי)
    if stu_city and site_city:
        city_component = 100 if stu_city == site_city else 0
    else:
        city_component = 50

    # ----- 3) בקשות מיוחדות – 45% -----
    # דוגמה: "קרוב" / "קרוב לבית"
    # אם ביקש קרוב והמוסד באותה עיר → 100, אחרת → 0
    # אם אין בקשה מיוחדת → 50
    if "קרוב" in stu_req:
        if stu_city and site_city and stu_city == site_city:
            special_component = 100
        else:
            special_component = 0
    else:
        special_component = 50

    parts = {
        "התאמת תחום": round(W.w_field * field_component),          # 50%
        "מרחק/גיאוגרפיה": round(W.w_city * city_component),        # 5%
        "בקשות מיוחדות": round(W.w_special * special_component),   # 45%
        "עדיפויות הסטודנט/ית": 0
    }

    score = int(np.clip(sum(parts.values()), 0, 100))
    return score, parts


# --- שיבוץ חמדני ---
def greedy_match(students_df: pd.DataFrame, sites_df: pd.DataFrame, W: Weights) -> pd.DataFrame:
    results = []
    supervisor_count = {}  # עד 2 סטודנטים למדריך (ניתן לשנות)

    for _, s in students_df.iterrows():
        cand = sites_df[sites_df["capacity_left"] > 0].copy()

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
                "_expl": {"התאמת תחום": 0, "מרחק/גיאוגרפיה": 0, "בקשות מיוחדות": 0, "עדיפויות הסטודנט/ית": 0}
            })
            continue

        # חישוב ציון
        def score_row(r):
            sc, parts = compute_score_with_explain(s, r, W)
            return pd.Series({"score": sc, "_parts": parts})

        cand[["score", "_parts"]] = cand.apply(score_row, axis=1)

        # הגבלת מדריכים (עד 2 סטודנטים)
        def allowed_supervisor(r):
            sup = r.get("שם המדריך", "")
            return supervisor_count.get(sup, 0) < 2

        filtered = cand[cand.apply(allowed_supervisor, axis=1)]

        if filtered.empty:
            # אם אין מדריך פנוי, בוחרים מהמוסדות עם capacity בלבד
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
                    "_expl": {"התאמת תחום": 0, "מרחק/גיאוגרפיה": 0, "בקשות מיוחדות": 0, "עדיפויות הסטודנט/ית": 0}
                })
                continue
            all_sites[["score", "_parts"]] = all_sites.apply(score_row, axis=1)
            filtered = all_sites.sort_values("score", ascending=False).head(1)
        else:
            filtered = filtered.sort_values("score", ascending=False)

        chosen = filtered.iloc[0]
        idx = chosen.name

        # עדכון קיבולת
        sites_df.at[idx, "capacity_left"] -= 1
        sup_name = chosen.get("שם המדריך", "")
        supervisor_count[sup_name] = supervisor_count.get(sup_name, 0) + 1

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

# ========= משתנים גלובליים פשוטים לדוחות =========
last_results_df = None
last_summary_df = None

@app.route("/", methods=["GET", "POST"])
def index():
    global last_results_df, last_summary_df

    context = {
        "results": None,
        "summary": None,
        "capacities": None,
        "expl_for_first": None,
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

            # טבלת תוצאות מרכזית
            df_show = pd.DataFrame({
                "אחוז התאמה": base_df["אחוז התאמה"].astype(int),
                "שם הסטודנט/ית": (base_df["שם פרטי"].astype(str) + " " + base_df["שם משפחה"].astype(str)).str.strip(),
                "תעודת זהות": base_df["ת\"ז הסטודנט"],
                "תחום התמחות": base_df["תחום ההתמחות במוסד"],
                "עיר המוסד": base_df["עיר המוסד"],
                "שם מקום ההתמחות": base_df["שם מקום ההתמחות"],
                "שם המדריך/ה": base_df["שם המדריך"],
            }).sort_values("אחוז התאמה", ascending=False)

            # דוח סיכום לפי מקום הכשרה
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

            # דוח קיבולות
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

            # הסבר ציון לשורה הראשונה (אפשר להרחיב ב-JS לפי בחירה)
            expl_for_first = None
            if len(base_df) > 0:
                first = base_df.iloc[0]
                parts = first["_expl"]
                expl_for_first = {
                    "student": f"{first['שם פרטי']} {first['שם משפחה']}",
                    "site": first["שם מקום ההתמחות"],
                    "score": int(first["אחוז התאמה"]),
                    "parts": parts
                }

            context.update({
                "results": df_show.to_dict(orient="records"),
                "summary": summary_df.to_dict(orient="records"),
                "capacities": cap_df.to_dict(orient="records"),
                "expl_for_first": expl_for_first,
                "error": None
            })

        except Exception as e:
            context["error"] = f"שגיאה במהלך השיבוץ: {e}"

    return render_template("index.html", **context)

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
    app.run(debug=True)

