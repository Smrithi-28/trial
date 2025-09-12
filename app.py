import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io
import base64
import folium
from folium.plugins import MarkerCluster
from scipy.stats import skew

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = "supersecretkey"

DATA_PATH = "cleaneddata.csv"
DEFAULT_K = 3

SAFE_LIMITS = {
    'Zn': 100, 'Fe': 100, 'Cu': 50, 'Mn': 100, 'B': 30, 'S': 50,
    'As': 10, 'Hg': 1, 'Se': 10, 'Co': 20, 'Mo': 50, 'Sb': 5,
    'V': 50, 'Tl': 1, 'Pb': 50
}

USERS = {
    "public": {"password": "public123", "role": "public", "email": "public@example.com"},
    "govt": {"password": "govt123", "role": "govt", "email": "govt@example.com"},
    "admin": {"password": "admin123", "role": "admin", "email": "admin@example.com"}
}

# Simple in-memory storage for demo (in production, use a database)
ISSUES = []

"""
Flask HMPI Dashboard Application
Refactored to use templates and static assets.
"""

# ----------------------
# Data Loader
# ----------------------
def load_and_prepare(df):
    if isinstance(df, str):
        df = pd.read_csv(df)
    df.columns = [c.strip() for c in df.columns]

    district_col = [c for c in df.columns if "district" in c.lower() or "site" in c.lower()][0]
    lat_col = [c for c in df.columns if "lat" in c.lower()][0]
    lon_col = [c for c in df.columns if "lon" in c.lower() or "long" in c.lower()][0]

    exclude = {district_col, lat_col, lon_col}
    metal_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]

    data = df.copy()
    for m in metal_cols:
        if data[m].isna().sum() > 0:
            s = skew(data[m].dropna()) if data[m].dropna().shape[0] > 0 else 0
            fill = data[m].median() if abs(s) > 1 else data[m].mean()
            data[m] = data[m].fillna(fill)

    # --- UPDATED HMPI CALCULATION (formula-based) ---
    # Step 1: Qi = (Mi / Si) * 100
      # --- Updated HMPI formula ---
    # Step 1: Compute Qi = (Mi / Si) * 100
    Qi = {}
    for m in metal_cols:
        safe_limit = SAFE_LIMITS.get(m, np.nan)
        if not np.isnan(safe_limit) and safe_limit > 0:
            Qi[m] = (data[m] / safe_limit) * 100
        else:
            Qi[m] = np.nan
    Qi_df = pd.DataFrame(Qi)

    # Step 2: Compute weights Wi = 1 / Si
       # --- Replace your HMPI calculation with actual HMPI formula ---
    weights = {}
    for m in metal_cols:
        limit = SAFE_LIMITS.get(m, None)
        if limit and limit > 0:
            weights[m] = 1.0 / limit
        else:
            weights[m] = 0

    # Sub-index Qi for each metal
    Q = {}
    for m in metal_cols:
        limit = SAFE_LIMITS.get(m, None)
        if limit and limit > 0:
            Q[m] = (data[m] / limit) * 100
        else:
            Q[m] = 0

    # HMPI = sum(Qi * Wi) / sum(Wi)
    numerator = sum(Q[m] * weights[m] for m in metal_cols if weights[m] > 0)
    denominator = sum(weights[m] for m in metal_cols if weights[m] > 0)
    data['HMPI'] = numerator / denominator if denominator != 0 else 0
    data['HMPI'] = data['HMPI'] / 100




    # ------------------------------------------------

    data['Site'] = data[district_col].astype(str)
    data['Latitude'] = pd.to_numeric(data[lat_col], errors='coerce')
    data['Longitude'] = pd.to_numeric(data[lon_col], errors='coerce')

    return data, district_col, metal_cols

try:
    DATA, DIST_COL, METAL_COLS = load_and_prepare(DATA_PATH)
except:
    DATA, DIST_COL, METAL_COLS = None, None, []

# ----------------------
# Plot helpers
# ----------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

def make_hmpi_hist(data):
    fig = plt.figure(figsize=(4,3))
    plt.hist(data['HMPI'].dropna(), bins=12)
    plt.xlabel('HMPI'); plt.ylabel('Count'); plt.title('HMPI distribution')
    return fig_to_base64(fig)

def make_cluster_pie(data):
    fig = plt.figure(figsize=(4,3))
    counts = data['Cluster'].value_counts().sort_index()
    plt.pie(counts.values, labels=[f'Cluster {i}' for i in counts.index], autopct='%1.1f%%')
    plt.title('Cluster share')
    return fig_to_base64(fig)

def build_map(data):
    if data.empty: return "<p>No map data</p>"
    center = [data['Latitude'].mean(), data['Longitude'].mean()]
    m = folium.Map(location=center, zoom_start=6)
    mc = MarkerCluster()
    for _, row in data.iterrows():
        if pd.isna(row['Latitude']) or pd.isna(row['Longitude']): continue
        h = row['HMPI']
        color = 'green' if h < 25 else 'yellow' if h < 50 else 'orange' if h < 75 else 'red'
        popup_html = f"<b>{row.get('Site','Site')}</b><br>HMPI: {h:.3f}<br>Cluster: {row.get('Cluster','-')}<br>"
        folium.CircleMarker(location=[row['Latitude'], row['Longitude']],
                            radius=6, color=color, fill=True, fill_opacity=0.8,
                            popup=popup_html).add_to(mc)
    mc.add_to(m)
    return m._repr_html_()

# ----------------------
# Routes
# ----------------------
@app.route("/", methods=["GET"]) 
def root():
    return redirect(url_for("landing"))

@app.route("/landing")
def landing():
    return render_template("landing.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        email = request.form.get("email")
        role = request.form.get("role", "public")
        
        if username in USERS:
            flash("Username already exists", "error")
            return render_template("signup.html")
        
        USERS[username] = {"password": password, "role": role, "email": email}
        flash("Account created successfully! Please login.", "success")
        return redirect(url_for("login"))
    
    return render_template("signup.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/dashboard", methods=["GET","POST"])
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    role = session.get("role", "public")

    uploaded_file = request.files.get("file")
    if uploaded_file and role=="govt":
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            temp_data, temp_dcol, temp_metals = load_and_prepare(df_uploaded)
            df = temp_data.copy()
            used_METAL_COLS = temp_metals
            DIST_COL_USED = temp_dcol
        except Exception as e:
            return f"<h3>File upload failed:</h3><pre>{str(e)}</pre>"
    else:
        df = DATA.copy()
        used_METAL_COLS = METAL_COLS
        DIST_COL_USED = DIST_COL

    selected = request.args.get("district", "_all_")
    if selected not in df[DIST_COL_USED].unique() and selected != "_all_":
        selected = "_all_"

    districts = sorted(df[DIST_COL_USED].unique())
    disp = df if selected=="_all_" else df[df[DIST_COL_USED]==selected].copy()

    k = int(request.args.get("k", DEFAULT_K))
    k = max(2, min(k, len(disp)))
    if len(disp) >= k:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        disp['Cluster'] = kmeans.fit_predict(disp[used_METAL_COLS].values)
    else:
        disp['Cluster'] = 0

    hist_b64 = make_hmpi_hist(disp)
    pie_b64 = make_cluster_pie(disp)
    map_html = build_map(disp)

    metal_summary = []
    for m in used_METAL_COLS:
        mean_val = round(disp[m].mean(), 2)
        safe_limit = SAFE_LIMITS.get(m, np.nan)
        status = "Safe" if mean_val <= safe_limit else "Unsafe"
        metal_summary.append((m, mean_val, safe_limit, status))

    return render_template(
        "dashboard.html",
        hist_b64=hist_b64,
        pie_b64=pie_b64,
        map_html=map_html,
        selected=selected,
        role=role,
        metal_summary=metal_summary,
        districts=districts,
    )



@app.route("/features")
def features():
    return render_template("features.html")

@app.route("/report", methods=["GET", "POST"])
def report_issue():
    if request.method == "POST":
        issue = {
            "id": len(ISSUES) + 1,
            "title": request.form.get("title"),
            "description": request.form.get("description"),
            "priority": request.form.get("priority"),
            "status": "Open",
            "reporter": session.get("username", "Anonymous"),
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        ISSUES.append(issue)
        flash("Issue reported successfully!", "success")
        return redirect(url_for("report_issue"))
    
    return render_template("report.html", issues=ISSUES)

@app.route("/admin/issues")
def admin_issues():
    if session.get("role") != "admin":
        flash("Access denied", "error")
        return redirect(url_for("dashboard"))
    return render_template("admin_issues.html", issues=ISSUES)

@app.route("/api/map-data")
def map_data():
    if DATA is None:
        return jsonify({"error": "No data available"})
    
    map_data = []
    for _, row in DATA.iterrows():
        if not pd.isna(row['Latitude']) and not pd.isna(row['Longitude']):
            map_data.append({
                "lat": float(row['Latitude']),
                "lng": float(row['Longitude']),
                "hmpi": float(row['HMPI']),
                "site": str(row.get('Site', 'Unknown')),
                "district": str(row.get(DIST_COL, 'Unknown')),
                "cluster": int(row.get('Cluster', 0))
            })
    
    return jsonify(map_data)

@app.route("/login", methods=["GET","POST"]) 
def login():
    error = None
    if request.method == "POST":
        uname = request.form.get("username")
        pwd = request.form.get("password")
        user = USERS.get(uname)
        if user and user["password"] == pwd:
            session["username"] = uname
            session["role"] = user["role"]
            return redirect(url_for("dashboard"))
        error = "Invalid credentials"
        flash(error, "danger")
    return render_template("login.html")



@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/general")
def general():
    if "username" not in session:
        return redirect(url_for("login"))
    df = DATA.copy() if DATA is not None else pd.DataFrame()
    map_html = build_map(df if not df.empty else pd.DataFrame(columns=["Latitude","Longitude","HMPI"]))
    table_html = df.head(500).to_html(classes='table table-striped table-bordered', index=False) if not df.empty else "<p>No data</p>"
    return render_template("general.html", map_html=map_html, table=table_html)

@app.route("/district/<district>")
def district_view(district):
    if "username" not in session:
        return redirect(url_for("login"))
    if DATA is None:
        return redirect(url_for("dashboard"))
    df = DATA.copy()
    if DIST_COL not in df.columns:
        return redirect(url_for("dashboard"))
    dist_df = df[df[DIST_COL] == district].copy()
    if dist_df.empty:
        flash("District not found", "warning")
        return redirect(url_for("dashboard"))
    avg_hmpi = round(dist_df['HMPI'].mean(), 3)
    metals_table = []
    for m in METAL_COLS:
        avg_val = round(dist_df[m].mean(), 2)
        limit = SAFE_LIMITS.get(m, np.nan)
        status = "Safe" if avg_val <= limit else "Unsafe"
        metals_table.append({"metal": m, "avg_val": avg_val, "limit": limit, "status": status})
    summary = {"avg_hmpi": avg_hmpi}
    return render_template("district.html", district=district, summary=summary, metals_table=metals_table)

@app.route("/site/<site>")
def site_view(site):
    if "username" not in session:
        return redirect(url_for("login"))
    if DATA is None:
        return redirect(url_for("dashboard"))
    df = DATA.copy()
    if 'Site' not in df.columns:
        return redirect(url_for("dashboard"))
    row = df[df['Site'].astype(str) == str(site)]
    if row.empty:
        flash("Site not found", "warning")
        return redirect(url_for("dashboard"))
    metals = METAL_COLS
    values = [float(row.iloc[0][m]) for m in metals]
    fig = plt.figure(figsize=(8,3))
    plt.bar(metals, values)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    chart_b64 = fig_to_base64(fig)
    return render_template("site.html", site=site, chart=chart_b64)

if __name__ == "__main__":
    app.run(debug=True, port=5002)
