import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template_string, request, redirect, url_for, session
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

app = Flask(__name__)
app.secret_key = "supersecretkey"

DATA_PATH = "cleaneddata.csv"
DEFAULT_K = 3

SAFE_LIMITS = {
    'Zn': 100, 'Fe': 100, 'Cu': 50, 'Mn': 100, 'B': 30, 'S': 50,
    'As': 10, 'Hg': 1, 'Se': 10, 'Co': 20, 'Mo': 50, 'Sb': 5,
    'V': 50, 'Tl': 1, 'Pb': 50
}

USERS = {
    "public": {"password": "public123", "role": "public"},
    "govt": {"password": "govt123", "role": "govt"}
}

# ----------------------
# Templates
# ----------------------
LOGIN_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Login - HMPI Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="container">
<h2 class="mt-4">HMPI Dashboard Login</h2>
<form method="post" class="mt-3">
  <div class="mb-3">
    <label>Username</label>
    <input type="text" name="username" class="form-control" required>
  </div>
  <div class="mb-3">
    <label>Password</label>
    <input type="password" name="password" class="form-control" required>
  </div>
  <button class="btn btn-primary">Login</button>
  {% if error %}<div class="text-danger mt-2">{{error}}</div>{% endif %}
</form>
</body>
</html>
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

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[metal_cols])
    scaled_df = pd.DataFrame(scaled, columns=metal_cols)
    data['HMPI'] = scaled_df.mean(axis=1)
    for c in metal_cols:
        data[f'scaled_{c}'] = scaled_df[c]

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
        color = 'green' if h < 0.25 else 'yellow' if h < 0.5 else 'orange' if h < 0.75 else 'red'
        popup_html = f"<b>{row.get('Site','Site')}</b><br>HMPI: {h:.3f}<br>Cluster: {row.get('Cluster','-')}<br>"
        folium.CircleMarker(location=[row['Latitude'], row['Longitude']],
                            radius=6, color=color, fill=True, fill_opacity=0.8,
                            popup=popup_html).add_to(mc)
    mc.add_to(m)
    return m._repr_html_()

# ----------------------
# Routes
# ----------------------
@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        uname = request.form.get("username")
        pwd = request.form.get("password")
        user = USERS.get(uname)
        if user and user["password"]==pwd:
            session["username"] = uname
            session["role"] = user["role"]
            return redirect(url_for("dashboard"))
        else:
            return render_template_string(LOGIN_TEMPLATE, error="Invalid credentials")
    return render_template_string(LOGIN_TEMPLATE, error=None)

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

    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HMPI Dashboard</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
body { padding: 20px; }
.chart-container { display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }
.chart-container img { flex: 1 1 45%; }
.map-container { height: 700px; width: 100%; border:1px solid #ddd; border-radius:8px; margin-bottom:20px; }
.metal-table { margin-top: 20px; overflow-x:auto; }
</style>
</head>
<body>
<div class="container">
<h2 class="mt-3 mb-3">HMPI Dashboard ({{ role }})</h2>
<div class="row mb-3">
{% if role=="govt" %}
<div class="col-md-4 mb-2">
<form method="POST" enctype="multipart/form-data">
<input type="file" name="file" class="form-control mb-1">
<button type="submit" class="btn btn-primary btn-sm w-100">Upload CSV</button>
</form>
</div>
{% endif %}
<div class="col-md-4 mb-2">
<form method="get">
<label for="district" class="form-label">Select District:</label>
<select name="district" id="district" class="form-select" onchange="this.form.submit()">
<option value="_all_">All Districts</option>
{% for d in districts %}
<option value="{{ d }}" {% if d == selected %}selected{% endif %}>{{ d }}</option>
{% endfor %}
</select>
</form>
</div>
<div class="col-md-4 mb-2 d-flex align-items-end">
<a href="{{ url_for('logout') }}" class="btn btn-secondary w-100">Logout</a>
</div>
</div>

<div class="chart-container mb-4">
<img src="data:image/png;base64,{{ hist_b64 }}" class="img-fluid rounded shadow">
<img src="data:image/png;base64,{{ pie_b64 }}" class="img-fluid rounded shadow">
</div>

<div class="map-container">
{{ map_html|safe }}
</div>

<div class="metal-table">
<h4>Metal Summary</h4>
<table class="table table-bordered table-striped">
<thead class="table-dark"><tr><th>Metal</th><th>Mean</th><th>Safe Limit</th><th>Status</th></tr></thead>
<tbody>
{% for metal, mean_val, safe_limit, status in metal_summary %}
<tr><td>{{ metal }}</td><td>{{ mean_val }}</td><td>{{ safe_limit }}</td><td>{{ status }}</td></tr>
{% endfor %}
</tbody>
</table>
</div>
</div>
</body>
</html>
""", hist_b64=hist_b64, pie_b64=pie_b64, map_html=map_html,
selected=selected, role=role, metal_summary=metal_summary, districts=districts)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
