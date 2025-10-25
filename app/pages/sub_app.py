# pages/02_quan_ly_cache_tuyen.py
from pathlib import Path
import json, math, datetime, os, shutil
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
# =============== CẤU HÌNH ===============
st.set_page_config(page_title="Xem tuyến (route_geoms / route_geoms1)", page_icon="🧭", layout="wide")
st.markdown("<h2 style='text-align:center;margin:0.25rem 0 1rem 0'>Xem tuyến từ CSV</h2>", unsafe_allow_html=True)

APP_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = APP_DIR / "results"
GEOMS_DIR_DEFAULT = RESULTS_DIR / "route_geoms"
GEOMS_DIR_NEW = RESULTS_DIR / "route_geoms1"
OSRM_URL = os.environ.get("OSRM_URL", "https://router.project-osrm.org")

# 🟢 CHỈNH LẠI ĐÂY CHO PHÙ HỢP CẤU TRÚC MỚI
PROJECT_ROOT = APP_DIR              # vì solver nằm trong app/
SOLVER_PKG = APP_DIR / "solver"     # đường dẫn tới app/solver
# Ép solver dùng OSRM public thay vì localhost:5000
# --- Force all OSRM calls to public server + fix lat,lon -> lon,lat ---
import re, requests as _rq

_OSRM_PUBLIC = "https://router.project-osrm.org"  # base host (không có /route hay /table)

# bản gốc
_orig_request = _rq.sessions.Session.request

_coord_pair_re = re.compile(r"(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)")

def _swap_if_latlon(x: str) -> str:
    """
    Với chuỗi 'a,b' nếu |a|<=90 và |b|<=180 thì coi là lat,lon -> trả 'b,a' (lon,lat).
    Ngược lại giữ nguyên.
    """
    try:
        a = float(x.group(1)); b = float(x.group(2))
        if abs(a) <= 90 and abs(b) <= 180:
            return f"{b},{a}"           # lon,lat
        return x.group(0)
    except Exception:
        return x.group(0)

def _fix_osrm_url(url: str) -> str:
    # 1) chuyển localhost:5000 / 127.0.0.1:5000 sang public OSRM
    if url.startswith("http://localhost:5000"):
        url = _OSRM_PUBLIC + url[len("http://localhost:5000"):]
    elif url.startswith("http://127.0.0.1:5000"):
        url = _OSRM_PUBLIC + url[len("http://127.0.0.1:5000"):]
    elif url.startswith("http://0.0.0.0:5000"):
        url = _OSRM_PUBLIC + url[len("http://0.0.0.0:5000"):]
    # 2) chỉ sửa toạ độ cho endpoint OSRM /route|/table
    if "/route/v1/driving/" in url or "/table/v1/driving/" in url:
        # tách query ?... để chỉ xử lý phần toạ độ
        if "?" in url:
            base, qs = url.split("?", 1)
            base = _coord_pair_re.sub(_swap_if_latlon, base)
            return f"{base}?{qs}"
        else:
            return _coord_pair_re.sub(_swap_if_latlon, url)
    return url

def _patched_request(self, method, url, *args, **kwargs):
    url = _fix_osrm_url(url)
    return _orig_request(self, method, url, *args, **kwargs)

# Áp dụng patch cho toàn bộ requests (bao gồm phần solver)
_rq.sessions.Session.request = _patched_request


def _import_solver():
    """
    Trả về hàm run_vrptw_advanced từ app/solver/vrptw_solver.py
    và tự xử lý import tuyệt đối 'from data_processing import ...'.
    """
    import sys, importlib, importlib.util

    # Đảm bảo app/solver có trong sys.path
    solver_path_str = str(SOLVER_PKG)
    if solver_path_str not in sys.path:
        sys.path.insert(0, solver_path_str)

    # Nạp trước data_processing (để solver import được)
    dp_path = SOLVER_PKG / "data_processing.py"
    if dp_path.exists():
        spec_dp = importlib.util.spec_from_file_location("data_processing", dp_path)
        dp_mod = importlib.util.module_from_spec(spec_dp)
        spec_dp.loader.exec_module(dp_mod)
        sys.modules["data_processing"] = dp_mod

    # Import vrptw_solver
    solver_file = SOLVER_PKG / "vrptw_solver.py"
    if not solver_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file solver: {solver_file}")

    spec = importlib.util.spec_from_file_location("vrptw_solver", solver_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Lấy hàm chạy chính
    if hasattr(mod, "run_vrptw_advanced"):
        return mod.run_vrptw_advanced
    elif hasattr(mod, "run_solver"):
        return mod.run_solver
    else:
        raise AttributeError("Không tìm thấy hàm run_vrptw_advanced hoặc run_solver trong vrptw_solver.py")




# đổi CWD tạm thời (nhiều solver dùng đường dẫn tương đối)
from contextlib import contextmanager
@contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)

def _run_solver_with_csv(csv_path: Path, time_limit: float):
    """
    Gọi solver với CSV đã chọn.
    - Solver nhận `customers_file` là basename (ví dụ temp_cus.csv)
    - Tạm chdir sang PROJECT_ROOT để tương thích đường dẫn tương đối
    """
    run_vrptw_advanced = _import_solver()
    customers_arg = os.path.basename(str(csv_path))
    with _pushd(PROJECT_ROOT):
        return run_vrptw_advanced(
            customers_file=customers_arg,
            max_iter=50000,
            time_limit=float(time_limit),
            use_cache=True,
        )

def _find_and_copy_solution_json() -> Path | None:
    """
    Tìm file solution JSON ở nhiều tên/đường dẫn, rồi copy về app/results/.
    Hỗ trợ các tên: solution_advanced.json, solution_demo.json.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    possible_names = {"solution_advanced.json", "solution_demo.json"}

    # 1) Nếu đã có sẵn ở đích
    for n in possible_names:
        p = RESULTS_DIR / n
        if p.exists():
            return p

    # 2) Các vị trí hay gặp
    candidates = []
    for n in possible_names:
        candidates += [
            PROJECT_ROOT / n,
            PROJECT_ROOT / "results" / n,
            SOLVER_PKG / n,
            SOLVER_PKG / "results" / n,
        ]

    # 3) Quét toàn dự án (lấy file mới nhất nếu có nhiều)
    newest, newest_mtime = None, -1
    try:
        for p in candidates + list(PROJECT_ROOT.rglob("solution_*.json")):
            if p and p.exists() and p.is_file():
                mt = p.stat().st_mtime
                if mt > newest_mtime:
                    newest, newest_mtime = p, mt
    except Exception:
        pass

    if newest:
        target = RESULTS_DIR / newest.name
        shutil.copy2(newest, target)
        return target

    return None



# =============== STATE ===============
if "cache_view_keys" not in st.session_state:
    st.session_state.cache_view_keys = []
if "last_run_keys" not in st.session_state:
    st.session_state.last_run_keys = []
if "selected_csv" not in st.session_state:
    st.session_state.selected_csv = None
if "active_geom_dir" not in st.session_state:
    st.session_state.active_geom_dir = str(GEOMS_DIR_DEFAULT)

# =============== TIỆN ÍCH ===============
def _human_size(n: int):
    if not n: return "0 B"
    units = ["B","KB","MB","GB","TB"]
    i = int(math.floor(math.log(max(n,1), 1024)))
    return f"{n/1024**i:.1f} {units[i]}"

def _read_distance_km(folder: Path, key: str):
    p = folder / f"{key}.distance"
    try:
        return float(p.read_text(encoding="utf-8").strip())
    except Exception:
        return None

def _list_cache_pairs(folder: Path) -> pd.DataFrame:
    if not folder.exists():
        return pd.DataFrame()
    files = list(folder.glob("*"))
    meta = {}
    for p in files:
        stem = p.stem
        if stem not in meta:
            meta[stem] = {
                "key": stem,
                "has_geojson": False, "geojson_size": 0, "geojson_mtime": None,
                "has_distance": False, "distance_size": 0, "distance_mtime": None,
            }
        if p.suffix.lower() == ".geojson":
            meta[stem]["has_geojson"] = True
            meta[stem]["geojson_size"] = p.stat().st_size
            meta[stem]["geojson_mtime"] = datetime.datetime.fromtimestamp(p.stat().st_mtime)
        elif p.suffix.lower() == ".distance":
            meta[stem]["has_distance"] = True
            meta[stem]["distance_size"] = p.stat().st_size
            meta[stem]["distance_mtime"] = datetime.datetime.fromtimestamp(p.stat().st_mtime)

    if not meta:
        return pd.DataFrame()
    df = pd.DataFrame(list(meta.values()))
    df["last_mtime"] = df[["geojson_mtime", "distance_mtime"]].max(axis=1)
    df["distance_km"] = [
        _read_distance_km(folder, k) if has else None
        for k, has in zip(df["key"], df["has_distance"])
    ]
    df = df.sort_values(by="last_mtime", ascending=False).reset_index(drop=True)
    df["route_name"] = df.index.map(lambda i: f"Route {i+1}")
    return df

def _coords_to_osrm_str(coords):  # coords = [(lat,lon),...]
    return ";".join(f"{lon},{lat}" for lat, lon in coords)

def _fetch_osrm_route(coords):
    """Trả về (geometry_geojson, distance_km) hoặc (None, None)."""
    import requests
    if len(coords) < 2: return None, None
    url = f"{OSRM_URL}/route/v1/driving/{_coords_to_osrm_str(coords)}"
    params = {"overview":"full","geometries":"geojson","steps":"false"}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        if data.get("routes"):
            g = data["routes"][0]["geometry"]
            dkm = data["routes"][0]["distance"]/1000.0
            return g, dkm
    except Exception as e:
        st.warning(f"OSRM lỗi: {e}")
    return None, None

def _load_csv_points(csv_path: Path) -> pd.DataFrame:
    """Đọc CSV theo định dạng temp_cus.csv (2 dòng header NUM_VEHICLE/CAPACITY có thể có)."""
    with open(csv_path, "r", encoding="utf-8") as f:
        first = f.readline()
        second = f.readline()
    skip = 2 if first.startswith("NUM_VEHICLE") or second.startswith("CAPACITY") else 0
    df = pd.read_csv(csv_path, skiprows=skip)
    # Chuẩn hoá tên cột
    if "lat" in df.columns and "Lat" not in df.columns: df["Lat"] = df["lat"]
    if "lon" in df.columns and "Long" not in df.columns: df["Long"] = df["lon"]
    if "id" in df.columns and "ID" not in df.columns: df["ID"] = df["id"]
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype(int)
    return df[["ID","Lat","Long"]].dropna()

def _build_route_geoms_from_solution(csv_path: Path, solution_json: Path, out_dir: Path):
    """Đọc solution_advanced.json -> tạo route_geoms1 (route_1/2/3...)."""
    points = _load_csv_points(csv_path)
    id2coord = {int(r.ID): (float(r.Lat), float(r.Long)) for _, r in points.iterrows()}
    if 0 not in id2coord:
        st.error("CSV thiếu depot (ID=0). Không thể tạo tuyến.")
        return False

    data = json.loads(solution_json.read_text(encoding="utf-8"))
    routes = data.get("routes", [])
    if not routes:
        st.error("solution_advanced.json không có dữ liệu routes.")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    for i, route_ids in enumerate(routes, start=1):
        coords = [id2coord[0]] + [id2coord[c] for c in route_ids if c in id2coord] + [id2coord[0]]
        geom, dkm = _fetch_osrm_route(coords)
        key = f"route_{i}"
        gj = out_dir / f"{key}.geojson"
        ds = out_dir / f"{key}.distance"
        if geom:
            gj.write_text(json.dumps(geom), encoding="utf-8")
        if dkm is not None:
            ds.write_text(f"{dkm}", encoding="utf-8")
        created += 1
    st.success(f"Đã tạo {created} tuyến trong: {out_dir}")
    return True

# =============== GIAO DIỆN ===============
left, right = st.columns([2,3])

with left:
    st.markdown("### 1) Chọn file CSV")
    UPLOAD_DIR = APP_DIR / "data" / "raw"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    available_csv = sorted([p.name for p in UPLOAD_DIR.glob("*.csv")])
    choice = st.radio("Nguồn dữ liệu:", ["📂 Chọn file có sẵn", "⬆️ Upload file mới"], horizontal=True)

    csv_path = None
    if choice == "📂 Chọn file có sẵn":
        if available_csv:
            default_idx = max(0, available_csv.index("test90.csv")) if "test90.csv" in available_csv else 0
            pick_name = st.selectbox("Chọn file CSV:", available_csv, index=default_idx)
            csv_path = UPLOAD_DIR / pick_name
            st.session_state.selected_csv = str(csv_path)
            st.success(f"Đã chọn file: {pick_name}")
        else:
            st.warning("⚠️ Chưa có file CSV nào trong `app/data/raw`.")
    else:
        up = st.file_uploader("Tải lên file CSV mới", type=["csv"])
        if up is not None:
            csv_path = UPLOAD_DIR / up.name
            with open(csv_path, "wb") as f:
                f.write(up.getbuffer())
            st.session_state.selected_csv = str(csv_path)
            st.success(f"Đã lưu file mới: {up.name}")

    if st.session_state.selected_csv:
        st.caption(f"📄 CSV đang chọn: `{st.session_state.selected_csv}`")

    # ==== Hành vi theo CSV ====
    active_dir = GEOMS_DIR_DEFAULT if (csv_path and csv_path.name.lower() == "test90.csv") else GEOMS_DIR_NEW
    st.session_state.active_geom_dir = str(active_dir)

    if csv_path and csv_path.name.lower() != "test90.csv":
        st.info("Bạn đang chọn CSV khác **test90.csv** → có thể chạy solver để tạo `route_geoms1`.")
        run_col1, run_col2 = st.columns([1,1])
        with run_col1:
            do_run = st.button("🔁 Chạy solver & tạo route_geoms1", use_container_width=True)
        with run_col2:
            time_limit = st.number_input("Giới hạn thời gian (giây)", min_value=20, step=10, value=60)

        if do_run:
            with st.spinner("Đang chạy solver ALNS và tạo route_geoms1…"):
                try:
                    _ = _run_solver_with_csv(csv_path, time_limit)  # gọi solver

                    solution_json = _find_and_copy_solution_json()
                    if not solution_json or not solution_json.exists():
                        st.error("Không tìm thấy solution JSON sau khi chạy solver. Kiểm tra lại đường lưu file trong solver.")
                    else:
                        ok = _build_route_geoms_from_solution(csv_path, solution_json, GEOMS_DIR_NEW)
                        if ok:
                            st.session_state.active_geom_dir = str(GEOMS_DIR_NEW)
                            keys_all = [p.stem for p in GEOMS_DIR_NEW.glob("*.geojson")]
                            st.session_state.cache_view_keys = keys_all
                            st.session_state.last_run_keys = keys_all
                            st.success("Đã sẵn sàng hiển thị các Route trong route_geoms1.")
                except Exception as e:
                    st.error(f"Lỗi khi chạy solver/tạo tuyến: {e}")

    # ==== Danh sách tuyến theo thư mục đang active ====
    active_dir = Path(st.session_state.active_geom_dir)
    st.markdown(f"### 2) Danh sách route trong `<code>{active_dir.name}</code>`", unsafe_allow_html=True)

    if st.button("👁️ Hiển thị TẤT CẢ route trong thư mục đang chọn", use_container_width=True):
        keys_all = [p.stem for p in active_dir.glob("*.geojson")]
        if keys_all:
            st.session_state.cache_view_keys = keys_all
            st.session_state.last_run_keys = keys_all
            st.success(f"Đã tải {len(keys_all)} route.")
        else:
            st.warning("Không tìm thấy *.geojson trong thư mục.")

    df = _list_cache_pairs(active_dir)
    if df.empty:
        st.warning(f"Chưa có file trong thư mục `{active_dir}`.")
        pick_names = []
    else:
        df_show = df.copy()
        df_show["geojson_size"] = df_show["geojson_size"].map(_human_size)
        df_show["distance_size"] = df_show["distance_size"].map(_human_size)
        df_show = df_show.rename(columns={
            "route_name":"Route", "key":"Key",
            "has_geojson":"GeoJSON?", "geojson_size":"GeoJSON Size",
            "geojson_mtime":"GeoJSON Modified", "has_distance":"Distance?",
            "distance_size":"Distance Size", "distance_mtime":"Distance Modified",
            "last_mtime":"Last Modified", "distance_km":"Distance (km)",
        })
        order = ["Route","Key","GeoJSON?","GeoJSON Size","GeoJSON Modified",
                 "Distance?","Distance Size","Distance Modified","Last Modified","Distance (km)"]
        df_show = df_show[order]
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        route_to_key = dict(zip(df["route_name"].tolist(), df["key"].tolist()))
        default_routes = [df.loc[df["key"] == k, "route_name"].iloc[0]
                          for k in st.session_state.get("last_run_keys", [])
                          if k in df["key"].values]
        pick_names = st.multiselect("Chọn route để xem/xoá:", list(route_to_key.keys()),
                                    default=default_routes, max_selections=30)

        c1, c2 = st.columns([1,1])
        disable_delete = (not pick_names) or (len(pick_names) == len(df))  # chặn xóa sạch
        view_btn = c1.button("👀 Xem trên bản đồ", use_container_width=True, disabled=not pick_names)
        del_btn  = c2.button("🗑️ Xoá route đã chọn", use_container_width=True, disabled=disable_delete)
        if len(pick_names) == len(df) and len(df) > 0:
            st.warning("Bạn đã chọn tất cả route — để an toàn, không cho phép xóa toàn bộ. Bỏ chọn bớt vài route để xóa.")

        pick_keys = [route_to_key[n] for n in pick_names]

        if view_btn and pick_keys:
            st.session_state.cache_view_keys = pick_keys
            st.session_state.last_run_keys = pick_keys

        if del_btn and pick_keys:
            ok, err = 0, 0
            for k in pick_keys:
                for ext in (".geojson", ".distance"):
                    p = active_dir / f"{k}{ext}"
                    if p.exists() and p.is_file():
                        try:
                            try: p.chmod(0o666)
                            except Exception: pass
                            p.unlink(); ok += 1
                        except Exception as e:
                            err += 1
                            st.error(f"Không xoá được {p.name}: {e}")
            st.success(f"Đã xoá {ok} file.{' Có lỗi ở ' + str(err) + ' file.' if err else ''}")
            st.session_state.cache_view_keys = []
            st.session_state.last_run_keys = []
            st.rerun()

# ===== KHU PHẢI: BẢN ĐỒ =====
with right:
    active_dir = Path(st.session_state.active_geom_dir)
    st.markdown(f"### 3) Bản đồ xem trước (đọc từ `{active_dir.name}`)")

    m = folium.Map(location=[21.0285,105.8542], zoom_start=12)
    to_draw = [k for k in st.session_state.get("cache_view_keys", []) if (active_dir / f"{k}.geojson").exists()]

    if to_draw:
        pal = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
               "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
        for i, k in enumerate(to_draw):
            gj = active_dir / f"{k}.geojson"
            color = pal[i % len(pal)]
            try:
                geom = json.loads(gj.read_text(encoding="utf-8"))
                folium.GeoJson(
                    data={"type":"Feature","geometry":geom},
                    style_function=lambda feat, col=color: {"color": col, "weight": 4, "opacity": 0.85}
                ).add_to(m)
                coords = geom.get("coordinates", [])
                if coords:
                    mid = coords[len(coords)//2]
                    dkm = _read_distance_km(active_dir, k)
                    route_name = f"Route {i+1}"
                    folium.Marker([mid[1], mid[0]],
                                  popup=f"{route_name} — {dkm:.2f} km" if dkm is not None else route_name
                                  ).add_to(m)
            except Exception as e:
                st.error(f"Lỗi đọc {gj.name}: {e}")
    else:
        st.info("Chưa có route để hiển thị. Chọn ở bảng bên trái hoặc bấm **'👁️ Hiển thị TẤT CẢ route trong thư mục đang chọn'**.")

    st_folium(m, height=620, use_container_width=True, key="map_cache_preview")

    if to_draw:
        sel_dist, total_km = [], 0.0
        for idx, k in enumerate(to_draw):
            dkm = _read_distance_km(active_dir, k)
            if dkm is not None:
                total_km += dkm
            sel_dist.append({"Route": f"Route {idx+1}", "Distance (km)": None if dkm is None else round(dkm, 2)})

        st.markdown("##### Tổng khoảng cách các route đang xem")
        st.dataframe(pd.DataFrame(sel_dist), use_container_width=True, hide_index=True)
        st.metric("Tổng quãng đường (km)", f"{total_km:.2f}")

st.markdown("<div style='text-align:center;margin-top:1rem'><a href='/' target='_self'>⬅️ Quay lại trang chính</a></div>", unsafe_allow_html=True)
