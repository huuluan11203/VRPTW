# app.py — VRPTW demo: Click map → Sửa trong bảng → Lưu CSV → Chạy solver → Vẽ tuyến + thống kê
from pathlib import Path
import os, json, random, time, importlib.util, sys, traceback, math
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, atan2

# =========================
# CẤU HÌNH & THƯ MỤC
# =========================
st.set_page_config(page_title='VRPTW Demo - Hà Nội', page_icon='🚚', layout='wide')
st.markdown(
    "<h1 style='text-align:center;margin-bottom:0.25rem'>VRPTW Demo – Định tuyến giao hàng có giới hạn thời gian</h1>"
    "<p style='text-align:center;color:#666'>Click map → chỉnh bảng → Lưu CSV → Chạy VRPTW → xem tuyến & ETA/ETD</p>",
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = APP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OSRM_URL = os.environ.get('OSRM_URL', 'https://router.project-osrm.org/route/v1/driving')

# =========================
# TIỆN ÍCH
# =========================
PALETTE = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
]
def route_color(idx: int) -> str:
    return PALETTE[(idx-1) % len(PALETTE)]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def coords_to_osrm_string(coords):
    return ";".join(f"{lon},{lat}" for lat, lon in coords)

# cache hình học route
CACHE_DIR = RESULTS_DIR / "route_geoms"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_route_from_osrm(coord_list, osrm_url=OSRM_URL, timeout=10):
    import requests
    if len(coord_list) < 2:
        return None, None
    key = str(abs(hash(tuple(coord_list))))
    cache_file_geom = CACHE_DIR / f"{key}.geojson"
    cache_file_dist = CACHE_DIR / f"{key}.distance"
    if cache_file_geom.exists() and cache_file_dist.exists():
        try:
            geom = json.loads(cache_file_geom.read_text(encoding='utf-8'))
            distance = float(cache_file_dist.read_text(encoding='utf-8'))
            return geom, distance
        except Exception:
            pass

    coord_str = coords_to_osrm_string(coord_list)
    url = f"{osrm_url}/{coord_str}"
    params = {"overview": "full", "geometries": "geojson", "steps": "false"}
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if "routes" in data and data["routes"]:
            route_data = data["routes"][0]
            geom = route_data["geometry"]
            distance_km = route_data["distance"] / 1000.0
            try:
                cache_file_geom.write_text(json.dumps(geom), encoding='utf-8')
                cache_file_dist.write_text(str(distance_km), encoding='utf-8')
            except Exception:
                pass
            return geom, distance_km
    except Exception as e:
        print(f"OSRM request error: {e}")
    return None, None

def add_geojson_route_to_map(m, geojson_geom, color="#3388ff", popup=None):
    if geojson_geom is None:
        return
    folium.GeoJson(
        data={"type": "Feature", "geometry": geojson_geom},
        style_function=lambda feat, col=color: {"color": col, "weight": 4, "opacity": 0.85}
    ).add_to(m)
    if popup:
        coords = geojson_geom.get('coordinates', [])
        if coords:
            mid = coords[len(coords)//2]
            folium.Marker([mid[1], mid[0]], popup=popup, opacity=0).add_to(m)

def calculate_real_route_costs(routes, customers_df, depot_coord):
    """Tính chi phí thực tế cho tất cả các route bằng OSRM; fallback Haversine."""
    id_to_latlon = {int(r.ID): (float(r.Lat), float(r.Long)) for _, r in customers_df.iterrows()}
    route_data, total_real_cost = [], 0.0
    for idx, route_ids in enumerate(routes, start=1):
        coords = [depot_coord] + [id_to_latlon[rid] for rid in route_ids if rid in id_to_latlon] + [depot_coord]
        geom, real_distance_km = get_route_from_osrm(coords)
        straight = 0.0
        for i in range(len(coords) - 1):
            straight += haversine_km(coords[i][0], coords[i][1], coords[i+1][0], coords[i+1][1])

        cost_km = real_distance_km if real_distance_km is not None else straight
        route_data.append({
            "route_idx": idx,
            "route_ids": route_ids,
            "coordinates": coords,
            "geometry": geom,
            "real_distance_km": real_distance_km,
            "straight_distance_km": straight,
            "final_distance_km": cost_km,
        })
        total_real_cost += cost_km
    return route_data, total_real_cost

# ===== DivIcon: nhãn STT trên map (đã phòng NaN) =====
def add_number_label(map_obj, lat, lon, text, color="#2b8a3e"):
    # chặn None/NaN/format lỗi
    if lat is None or lon is None:
        return
    try:
        lat = float(lat); lon = float(lon)
    except (TypeError, ValueError):
        return
    if math.isnan(lat) or math.isnan(lon):
        return

    folium.Marker(
        [lat, lon],
        icon=folium.DivIcon(
            html=f"""
            <div style="
                background:#ffffff; border:2px solid {color};
                color:{color}; font-weight:700; font-size:12px;
                width:24px; height:24px; line-height:22px; text-align:center;
                border-radius:50%; box-shadow:0 0 4px rgba(0,0,0,0.25);
            ">{text}</div>
            """,
            icon_size=(24,24),
            icon_anchor=(12,12),
        ),
    ).add_to(map_obj)

# --- lưu CSV tạm (có depot ID=0) ---
def build_temp_csv(df_points: pd.DataFrame, depot_xy: tuple, num_vehicles: int, capacity: int) -> Path:
    out = RAW_DIR / "temp_cus.csv"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    need_src = ["ID","Lat","Long","demand","ready_time","due_time","service_time"]
    df = df_points.copy()
    for c in need_src:
        if c not in df.columns:
            df[c] = 0

    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").fillna(0).astype(int)
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["Long"] = pd.to_numeric(df["Long"], errors="coerce")
    for c in ["demand","ready_time","due_time","service_time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    depot_row = pd.DataFrame([{
        "ID": 0, "Lat": float(depot_xy[0]), "Long": float(depot_xy[1]),
        "demand": 0, "ready_time": 0, "due_time": 1440, "service_time": 0,
    }])
    df = pd.concat([depot_row, df[need_src]], ignore_index=True)

    df_out = df.rename(columns={
        "Lat": "lat", "Long": "lon",
        "demand": "DEMAND", "ready_time": "READY_TIME",
        "due_time": "DUE_DATE", "service_time": "SERVICE_TIME",
    })[["ID","lat","lon","DEMAND","READY_TIME","DUE_DATE","SERVICE_TIME"]]

    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write(f"NUM_VEHICLE={int(num_vehicles)}\n")
        f.write(f"CAPACITY={int(capacity)}\n")
        df_out.to_csv(f, index=False, lineterminator="\n")
    return out

def _validate_points(df_points: pd.DataFrame):
    msgs = []
    df = df_points.copy()
    df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
    df["Long"] = pd.to_numeric(df["Long"], errors="coerce")
    df = df[df["Lat"].between(-90, 90)]
    df = df[df["Long"].between(-180, 180)]
    for c in ["demand","ready_time","due_time","service_time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    bad_tw = df["due_time"] < df["ready_time"]
    if bad_tw.any():
        msgs.append(f"Đã tự sửa {bad_tw.sum()} dòng có due_time < ready_time (set bằng ready_time).")
        df.loc[bad_tw, "due_time"] = df.loc[bad_tw, "ready_time"]
    df = df.drop_duplicates(subset=["Lat","Long","ready_time","due_time","demand","service_time"])
    df = df.dropna(subset=["Lat","Long"])  # đảm bảo không còn NaN toạ độ
    if len(df) == 0:
        msgs.append("Không còn khách hợp lệ sau khi lọc.")
    return df, msgs

def load_customers_for_display():
    p = RAW_DIR / "temp_cus.csv"
    if not p.exists():
        return pd.DataFrame()
    lines = p.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, skiprows=2)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "Lat" not in df.columns and "lat" in df.columns:
        df["Lat"] = pd.to_numeric(df["lat"], errors="coerce")
    if "Long" not in df.columns and "lon" in df.columns:
        df["Long"] = pd.to_numeric(df["lon"], errors="coerce")
    if "ID" not in df.columns and "id" in df.columns:
        df["ID"] = pd.to_numeric(df["id"], errors="coerce").astype(int)
    else:
        df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype(int)
    df = df.dropna(subset=["Lat","Long"])
    return df

def load_solution_demo():
    p = RESULTS_DIR / "solution_demo.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

# =========================
# STATE & SIDEBAR
# =========================
if "picked" not in st.session_state:
    st.session_state.picked = []

if "show_result" not in st.session_state:
    st.session_state.show_result = False

# Buffer DataFrame để chỉnh sửa mượt (không mất khi rerun)
if "picked_df" not in st.session_state:
    st.session_state.picked_df = pd.DataFrame(st.session_state.picked) if st.session_state.picked else pd.DataFrame(
        columns=["ID","Lat","Long","demand","ready_time","due_time","service_time"]
    )

st.sidebar.header("Thiết lập")
default_center = (21.0285, 105.8542)  # Hà Nội
solver_time = st.sidebar.number_input("Giới hạn thời gian solver (giây)", min_value=10, step=10, value=60)

st.sidebar.markdown("---")
st.sidebar.subheader("Depot")
depot_lat = st.sidebar.number_input("Depot Lat", value=default_center[0], format="%.6f")
depot_lon = st.sidebar.number_input("Depot Lon", value=default_center[1], format="%.6f")
depot_coord = (float(depot_lat), float(depot_lon))

st.sidebar.markdown("---")
st.sidebar.subheader("Mặc định cho điểm mới")
col_d, col_s = st.sidebar.columns(2)
with col_d:
    default_demand = st.number_input("Demand", min_value=0, value=1)
with col_s:
    default_service = st.number_input("Service time", min_value=0, value=0)
col_t1, col_t2 = st.sidebar.columns(2)
with col_t1:
    default_ready = st.number_input("Ready time", min_value=0, value=0)
with col_t2:
    default_due = st.sidebar.number_input("Due time", min_value=0, value=1440)

st.sidebar.markdown("---")
st.sidebar.subheader("Ràng buộc VRPTW")
num_vehicles = st.sidebar.number_input("Số xe tối đa", min_value=1, value=10, step=1)
capacity = st.sidebar.number_input("Tải trọng xe (đơn vị demand)", min_value=1, value=50, step=1)

st.sidebar.markdown("---")
add_mode = st.sidebar.toggle("➕ Chế độ thêm điểm từ bản đồ", value=True)

# init state cho chống double-click
if "last_click" not in st.session_state:
    st.session_state.last_click = {"lat": None, "lon": None, "ts": 0.0}

# Nút mở trang "Quản lý cache tuyến"
if st.button("🧭 Quản lý cache tuyến (route_geoms)"):
    st.switch_page("pages/sub_app.py")   # ✅ đúng

# =========================
# 1) BẢN ĐỒ – CLICK ĐỂ THÊM KHÁCH
# =========================
st.markdown("#### Hãy chọn địa điểm trên bản đồ")
m = folium.Map(location=depot_coord, zoom_start=12)
folium.Marker(
    depot_coord, popup="Depot (ID=0)",
    icon=folium.Icon(color="red", icon="home", prefix='fa')
).add_to(m)

# các điểm đã chọn (đánh STT) — đã phòng NaN
for idx, row in enumerate(st.session_state.picked, start=1):
    lat = row.get("Lat"); lon = row.get("Long")
    # bỏ qua nếu thiếu hoặc NaN
    try:
        lat = float(lat); lon = float(lon)
    except (TypeError, ValueError):
        continue
    if math.isnan(lat) or math.isnan(lon):
        continue

    add_number_label(m, lat, lon, text=str(idx), color="#2b8a3e")
    folium.Marker([lat, lon], opacity=0).add_child(
        folium.Popup(f'#{idx} — ID {row.get("ID","?")} (d={row.get("demand","?")})')
    ).add_to(m)

# gợi ý trực quan click
m.add_child(folium.LatLngPopup())
output = st_folium(m, height=520, use_container_width=True, key="map_pick", debug=False)

if add_mode and output and output.get("last_clicked"):
    lat = float(output["last_clicked"]["lat"]); lon = float(output["last_clicked"]["lng"])
    now = time.time()
    prev = st.session_state.last_click
    same_spot = (prev["lat"] is not None and abs(prev["lat"]-lat) < 1e-6 and abs(prev["lon"]-lon) < 1e-6)
    fast = (now - prev["ts"] < 0.4)
    if not (same_spot or fast):
        next_id = 1 if not st.session_state.picked else max(p.get("ID", 0) for p in st.session_state.picked) + 1
        st.session_state.picked.append({
            "ID": next_id, "Lat": lat, "Long": lon,
            "demand": int(default_demand), "ready_time": int(default_ready),
            "due_time": int(default_due), "service_time": int(default_service),
        })
        # cập nhật buffer
        st.session_state.picked_df = pd.DataFrame(st.session_state.picked).reset_index(drop=True)
        st.session_state.last_click = {"lat": lat, "lon": lon, "ts": now}

# =========================
# 2) BẢNG CHỈNH SỬA (mượt)
# =========================
st.markdown("#### Danh sách điểm đã chọn")

def _normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy(); df["ID"] = range(1, len(df) + 1)
    return df

if not st.session_state.picked_df.empty:
    with st.form("edit_points_form", clear_on_submit=False):
        edited = st.data_editor(
            st.session_state.picked_df,
            key="picked_editor",
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "ID": st.column_config.NumberColumn(help="Mã khách (sẽ tự chuẩn hoá)"),
                "Lat": st.column_config.NumberColumn(format="%.6f"),
                "Long": st.column_config.NumberColumn(format="%.6f"),
                "demand": st.column_config.NumberColumn(min_value=0, step=1),
                "ready_time": st.column_config.NumberColumn(min_value=0, step=1),
                "due_time": st.column_config.NumberColumn(min_value=0, step=1),
                "service_time": st.column_config.NumberColumn(min_value=0, step=1),
            }
        )
        c1, c2, c3 = st.columns([1,1,6])
        apply_btn = c1.form_submit_button("Áp dụng chỉnh sửa", type="primary", use_container_width=True)
        del_last = c2.form_submit_button("Xoá điểm cuối", use_container_width=True)
        clear_all = c3.form_submit_button("Xoá tất cả", use_container_width=False)

    if apply_btn:
        # --- Làm sạch dữ liệu để tránh NaN khi vẽ map ---
        df_new = edited.copy()

        # ép kiểu về số; giá trị không hợp lệ -> NaN
        for c in ["Lat", "Long", "demand", "ready_time", "due_time", "service_time"]:
            df_new[c] = pd.to_numeric(df_new[c], errors="coerce")

        # loại các hàng thiếu toạ độ
        before = len(df_new)
        df_new = df_new.dropna(subset=["Lat", "Long"])
        after = len(df_new)
        if after < before:
            st.warning(f"Đã loại {before-after} dòng do thiếu Lat/Long.")

        # điền mặc định cho các cột số còn NaN
        for c in ["demand", "ready_time", "due_time", "service_time"]:
            df_new[c] = df_new[c].fillna(0).astype(int)

        # chuẩn hoá ID theo thứ tự mới
        df_new = df_new[["ID","Lat","Long","demand","ready_time","due_time","service_time"]]
        df_new = _normalize_ids(df_new).reset_index(drop=True)

        st.session_state.picked_df = df_new
        st.session_state.picked = df_new.to_dict(orient="records")
        st.success("Đã áp dụng chỉnh sửa.")
    elif del_last:
        if not st.session_state.picked_df.empty:
            st.session_state.picked_df = st.session_state.picked_df.iloc[:-1].reset_index(drop=True)
            st.session_state.picked = st.session_state.picked_df.to_dict(orient="records")
            st.success("Đã xoá điểm cuối.")
    elif clear_all:
        st.session_state.picked_df = st.session_state.picked_df.iloc[0:0]
        st.session_state.picked = []
        st.success("Đã xoá tất cả điểm.")
else:
    st.info("Chưa có điểm nào. Hãy click lên bản đồ để thêm khách hàng.")

# =========================
# 3) LƯU CSV & CHẠY SOLVER
# =========================
st.markdown("#### Lưu danh sách & Chạy VRPTW")
save_col, run_col = st.columns([1,1])
with save_col:
    save_csv = st.button("Lưu địa điểm", type="primary", use_container_width=True)
with run_col:
    do_run = st.button("Chạy VRPTW", use_container_width=True)

solution_path = RESULTS_DIR / "solution_demo.json"

# 3.1 LƯU CSV
if save_csv:
    if st.session_state.picked_df.empty:
        st.warning("Chưa có điểm nào để lưu.")
    else:
        raw_df = st.session_state.picked_df.copy()
        df_points, notes = _validate_points(raw_df)
        if len(df_points) == 0:
            st.error("Danh sách rỗng hoặc không hợp lệ. Kiểm tra lại bảng (Lat/Long, time window...).")
        else:
            if notes:
                for m in notes: st.warning(m)
            csv_path = build_temp_csv(df_points, depot_coord, num_vehicles, capacity)
            st.success(f"Đã lưu {csv_path}")
            st.stop()

# 3.2 CHẠY SOLVER
if do_run:
    p = RAW_DIR / "temp_cus.csv"
    if not p.exists():
        st.error("Chưa có data/raw/temp_cus.csv. Hãy bấm **Lưu danh sách (CSV)** trước.")
        st.stop()
    with st.spinner("Đang chạy VRPTW (ALNS)…"):
        try:
            SOLVER_FILE = APP_DIR / "solver" / "vrptw_solver.py"
            if not SOLVER_FILE.exists():
                raise FileNotFoundError(f"Không tìm thấy: {SOLVER_FILE}")
            sys.path.insert(0, str(SOLVER_FILE.parent))
            spec = importlib.util.spec_from_file_location("vrptw_solver", str(SOLVER_FILE))
            vrptw_solver = importlib.util.module_from_spec(spec)
            sys.modules["vrptw_solver"] = vrptw_solver
            spec.loader.exec_module(vrptw_solver)

            # Patch: compute_time_matrix_OSRM resilient
            import numpy as np, requests as _rq
            def _haversine_km(a, b):
                (lat1, lon1), (lat2, lon2) = a, b
                R = 6371.0088
                dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
                s = (sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2)
                return 2 * R * atan2(sqrt(s), sqrt(1 - s))
            def _compute_time_matrix_resilient(locations, speed_kmh=30.0):
                base = "https://router.project-osrm.org/table/v1/driving/"
                coords = ";".join([f"{lon},{lat}" for (lat, lon) in locations])
                url = base + coords
                try:
                    r = _rq.get(url, params={"annotations": "duration"}, timeout=20)
                    r.raise_for_status()
                    data = r.json()
                    durs = data.get("durations")
                    if durs:
                        n = len(durs)
                        for i in range(n):
                            for j in range(n):
                                if durs[i][j] is None:
                                    durs[i][j] = 0 if i == j else int(1e9)
                        return [[int(round(x)) for x in row] for row in durs]
                except Exception:
                    pass
                n = len(locations); M = np.zeros((n, n), dtype=int)
                speed_mps = speed_kmh * 1000.0 / 3600.0
                for i in range(n):
                    for j in range(n):
                        if i == j: M[i, j] = 0
                        else:
                            dist_km = _haversine_km(locations[i], locations[j])
                            M[i, j] = int(round((dist_km * 1000.0) / speed_mps))
                return M.tolist()
            vrptw_solver.compute_time_matrix_OSRM = _compute_time_matrix_resilient

            best_solution, best_cost = vrptw_solver.run_vrptw_advanced(
                customers_file=os.path.basename(str(p)),
                max_iter=50000,
                time_limit=float(solver_time),
                use_cache=True,
            )

            adv_json = APP_DIR / "results" / "solution_advanced.json"
            if adv_json.exists():
                solution_data = json.loads(adv_json.read_text(encoding="utf-8"))
            else:
                solution_data = {
                    "cost": float(best_cost),
                    "num_routes": len(best_solution),
                    "routes": best_solution
                }
            solution_path.write_text(json.dumps(solution_data, indent=2, ensure_ascii=False), encoding="utf-8")
            st.success(f"Đã lưu kết quả vào {solution_path}")
            st.session_state.show_result = True
        except Exception:
            st.error("Lỗi khi chạy solver:")
            st.code(traceback.format_exc())

# ===== Helpers cho mục 4 (cache data) =====
@st.cache_data(show_spinner=False)
def _compute_route_data_cached(routes_sig, cust_sig, depot_coord):
    routes = [list(r) for r in routes_sig]
    customers_df = pd.DataFrame(
        [{"ID": i, "Lat": lat, "Long": lon} for (i, lat, lon) in cust_sig]
    )
    return calculate_real_route_costs(routes, customers_df, depot_coord)

def _signatures_for_cache(solution, customers, depot):
    routes_sig = tuple(tuple(r) for r in solution.get("routes", []))
    cust_sig = tuple((int(r.ID), float(r.Lat), float(r.Long)) for _, r in customers.iterrows())
    depot_sig = (float(depot[0]), float(depot[1]))
    return routes_sig, cust_sig, depot_sig

# =============================
# 4) Kết quả tuyến & thống kê
# =============================
st.markdown("#### Kết quả tuyến & thống kê")

if not st.session_state.get("show_result"):
    st.info("Nhấn ** Chạy VRPTW** để xem bản đồ & bảng kết quả.")
else:
    customers = load_customers_for_display()
    solution = load_solution_demo()

    if customers.empty:
        st.info("Chưa có `data/raw/temp_cus.csv`. Hãy Lưu CSV trước.")
    elif not solution or "routes" not in solution or not solution["routes"]:
        st.info("Chưa có kết quả `solution_demo.json` (hoặc rỗng).")
    else:
        try:
            depot_row = customers.loc[customers["ID"] == 0].iloc[0]
        except Exception:
            st.error("Không tìm thấy depot (ID=0) trong dữ liệu khách hàng.")
        else:
            depot_coord_show = (float(depot_row["Lat"]), float(depot_row["Long"]))

            # Tính quãng đường thực tế (OSRM)
            routes_sig, cust_sig, depot_sig = _signatures_for_cache(solution, customers, depot_coord_show)
            with st.spinner("Đang tính khoảng cách thực tế (OSRM)…"):
                route_data, total_real_cost = _compute_route_data_cached(routes_sig, cust_sig, depot_sig)

            # Bảng thống kê tuyến
            st.subheader("Bảng thống kê tuyến")
            display_data = []
            for d in route_data:
                if d.get('real_distance_km') is not None:
                    dist_txt = f"{d.get('final_distance_km', d.get('real_distance_km', 0.0)):.2f}"
                else:
                    dist_txt = f"{d.get('straight_distance_km', d.get('final_distance_km', 0.0)):.2f}*"
                display_data.append({
                    "Route #": d["route_idx"],
                    "Số điểm": len(d["route_ids"]),
                    "Điểm đi qua": " → ".join(str(x) for x in d["route_ids"]),
                    "Khoảng cách (km)": dist_txt,
                })
            df_routes = pd.DataFrame(display_data)
            st.dataframe(df_routes, use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            c1.metric("Tổng số tuyến", len(solution["routes"]))
            c2.metric("Tổng chi phí (km)", f"{total_real_cost:.2f}")

            if any(d.get('real_distance_km') is None for d in route_data):
                st.info("Một số tuyến dùng khoảng cách ước tính (đường thẳng) do không kết nối được OSRM (đánh dấu *).")

            # ===== ETA/ETD theo ready_time / service_time =====
            SPEED_KMH = 30.0  # có thể đưa lên sidebar nếu muốn
            def _col(df, up, low): return up if up in df.columns else (low if low in df.columns else None)
            _ready_col = _col(customers, "READY_TIME", "ready_time")
            _due_col   = _col(customers, "DUE_DATE",    "due_time")
            _serv_col  = _col(customers, "SERVICE_TIME","service_time")
            id2tw = {}
            for _, r in customers.iterrows():
                rid = int(r["ID"])
                ready = int(r[_ready_col]) if _ready_col else 0
                due   = int(r[_due_col])   if _due_col   else 1440
                serv  = int(r[_serv_col])  if _serv_col  else 0
                id2tw[rid] = (ready, due, serv)
            def _fmt_hhmm(m):
                m = int(round(m)); h = m // 60; mm = m % 60
                return f"{h:02d}:{mm:02d}"
            def _travel_minutes_km(lat1, lon1, lat2, lon2, speed_kmh=SPEED_KMH):
                d_km = haversine_km(lat1, lon1, lat2, lon2)
                t_min = (d_km / speed_kmh) * 60.0
                return d_km, t_min

            st.subheader("Chi tiết thời gian")
            tabs = st.tabs([f"Tuyến {d['route_idx']}" for d in route_data])
            for tab, d in zip(tabs, route_data):
                with tab:
                    coords = d["coordinates"][:]
                    stop_ids = [0] + d["route_ids"][:]
                    if stop_ids[-1] != 0:
                        stop_ids = stop_ids + [0]

                    rows = []; depot_ready = id2tw.get(0, (0, 1440, 0))[0]; cur_time = float(depot_ready)
                    total_drive_km = 0.0
                    for leg_idx in range(len(stop_ids)-1):
                        sid = stop_ids[leg_idx]; nid = stop_ids[leg_idx+1]
                        (lat1, lon1) = coords[leg_idx]; (lat2, lon2) = coords[leg_idx+1]
                        d_km, drive_min = _travel_minutes_km(lat1, lon1, lat2, lon2)
                        total_drive_km += d_km
                        eta = cur_time + drive_min
                        ready, due, serv = id2tw.get(nid, (0,1440,0))
                        wait = max(0.0, ready - eta)
                        start_service = eta + wait
                        etd = start_service + serv
                        violation = "" if eta <= due else f"🔴 trễ cửa sổ (đến {int(eta)} > due {due})"
                        rows.append({
                            "Leg": f"{sid} → {nid}",
                            "Di chuyển (km)": round(d_km, 2),
                            "Di chuyển (phút)": round(drive_min, 1),
                            "ETA (đến nơi)": _fmt_hhmm(eta),
                            "Wait (phút)": round(wait, 1),
                            "Bắt đầu phục vụ": _fmt_hhmm(start_service),
                            "Service (phút)": serv,
                            "ETD (rời đi)": _fmt_hhmm(etd),
                            "TW đích": f"[{_fmt_hhmm(ready)}–{_fmt_hhmm(due)}]",
                            "Vi phạm": violation,
                        })
                        cur_time = etd
                    df_eta = pd.DataFrame(rows)
                    st.dataframe(df_eta, use_container_width=True, hide_index=True)
                    total_drive_min = sum(r["Di chuyển (phút)"] for r in rows)
                    st.caption(
                        f"⏱️ Tổng thời gian di chuyển ≈ {round(total_drive_min,1)} phút "
                        f"(~{_fmt_hhmm(total_drive_min)}), tổng quãng đường ≈ {round(total_drive_km,2)} km "
                        f"— tốc độ giả định {SPEED_KMH:g} km/h."
                    )

            # ===== VẼ BẢN ĐỒ KẾT QUẢ (đánh STT) =====
            with st.expander("Chẩn đoán dữ liệu bản đồ", expanded=False):
                n_routes = len(route_data)
                n_missing_coords = sum(1 for d in route_data if not d.get("coordinates"))
                n_missing_geom = sum(1 for d in route_data if not d.get("geometry"))
                st.write({"Tổng số tuyến": n_routes,
                          "Số tuyến thiếu coordinates": n_missing_coords,
                          "Số tuyến thiếu geometry": n_missing_geom})
                st.write("Ví dụ 1–2 tuyến đầu:", route_data[:2])

            _id2coord = {int(row["ID"]): (float(row["Lat"]), float(row["Long"])) for _, row in customers.iterrows()}

            def _ensure_coordinates(d, depot_coord_show):
                if d.get("coordinates"):
                    return d["coordinates"]
                coords = [depot_coord_show]
                for cid in d.get("route_ids", []):
                    if cid in _id2coord:
                        coords.append(_id2coord[cid])
                coords.append(depot_coord_show)
                return coords

            def _build_result_map(route_data, depot_coord_show):
                mm = folium.Map(location=depot_coord_show, zoom_start=12)
                folium.Marker(
                    depot_coord_show,
                    popup="Depot (ID=0)",
                    icon=folium.Icon(color="red", icon="home", prefix='fa')
                ).add_to(mm)

                for d in route_data:
                    color = route_color(d["route_idx"])
                    coords = _ensure_coordinates(d, depot_coord_show)

                    if d.get("geometry"):
                        add_geojson_route_to_map(
                            mm, d["geometry"], color=color,
                            popup=f"Route #{d['route_idx']} — {d.get('final_distance_km', 0.0):.2f} km"
                        )
                    else:
                        if len(coords) >= 2:
                            folium.PolyLine(
                                coords, color=color, weight=3, opacity=0.9,
                                popup=f"Route #{d['route_idx']} — {d.get('final_distance_km', 0.0):.2f} km"
                            ).add_to(mm)

                    # Nhãn STT cho từng điểm dừng (bỏ depot đầu/cuối)
                    for seq, (lat, lon) in enumerate(coords[1:-1], start=1):
                        ridx = seq - 1
                        cust_id = d["route_ids"][ridx] if 0 <= ridx < len(d.get("route_ids", [])) else ''
                        add_number_label(mm, lat, lon, text=str(seq), color=color)
                        folium.Marker([lat, lon], opacity=0).add_child(
                            folium.Popup(f"Route {d['route_idx']} - Điểm {seq} - ID {cust_id}")
                        ).add_to(mm)
                return mm

            mm = _build_result_map(route_data, depot_coord_show)
            _ = st_folium(mm, height=620, use_container_width=True, key="map_result")

st.markdown("---")
st.caption("Flow: Click map → sửa bảng → Lưu CSV → Chạy VRPTW → xem tuyến.")
