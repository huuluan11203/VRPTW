# app_simple_routes.py - Hiển thị toàn bộ các route từ file kết quả, vẽ bản đồ & tính chi phí THỰC TẾ

from pathlib import Path
import math
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
from math import radians, sin, cos, sqrt, atan2
import random
import requests
import json
import os

st.set_page_config(page_title='Định tuyến xe - Hà Nội', page_icon='🚚', layout='wide')
st.title('Định tuyến xe giao hàng – Hà Nội')
st.caption('Hiển thị các route từ file kết quả định tuyến và tính chi phí từng tuyến.')

# --------------------------------
# HÀM HỖ TRỢ
# --------------------------------
def _try_paths(filename):
    """Try a set of sensible paths relative to the app directory.

    Looks in ./data/raw, ./data, current working dir, and /mnt/data.
    Returns a Path or None.
    """
    base = Path(__file__).parent
    candidates = [
        base / 'data' / 'raw' / filename,
        base / 'data' / filename,
        base / filename,
        Path(filename),
        Path('/mnt/data') / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def random_color():
    return f"#{random.randint(0, 0xFFFFFF):06x}"

# OSRM / routing settings
OSRM_URL = os.environ.get('OSRM_URL', 'http://localhost:5000/route/v1/driving')
CACHE_DIR = Path(__file__).parent / 'results' / 'route_geoms'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def coords_to_osrm_string(coords):
    # coords: list of (lat, lon)
    return ";".join(f"{lon},{lat}" for lat, lon in coords)

def get_route_from_osrm(coord_list, osrm_url=OSRM_URL, timeout=10):
    """Request OSRM for a route geometry (GeoJSON LineString) và distance thực tế.

    Returns (geojson_geometry, distance_km) or (None, None) on failure.
    """
    if len(coord_list) < 2:
        return None, None

    # Use a stable cache key based on coordinates
    key = str(abs(hash(tuple(coord_list))))
    cache_file_geom = CACHE_DIR / f"{key}.geojson"
    cache_file_dist = CACHE_DIR / f"{key}.distance"
    
    # Check cache for both geometry and distance
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
            distance_meters = route_data["distance"]
            distance_km = distance_meters / 1000.0
            
            # Cache both geometry and distance
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
            folium.Marker([mid[1], mid[0]], popup=popup).add_to(m)

# --------------------------------
# ĐỌC FILE KHÁCH HÀNG
# --------------------------------
@st.cache_data
def load_customers(input_file: str):
    p = _try_paths(input_file)
    if not p:
        st.error("Không tìm thấy file customers.csv trong thư mục data/raw hoặc data.")
        return pd.DataFrame()

    # Handle files that may contain initial key=value lines before CSV header
    header_idx = None
    with open(p, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if line.strip().upper().startswith('ID,'):
                header_idx = i
                break

    if header_idx is None:
        df = pd.read_csv(p, dtype=str)
    else:
        df = pd.read_csv(p, skiprows=header_idx)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Ensure ID exists
    if 'ID' not in df.columns:
        st.error('File customers.csv phải chứa cột ID')
        return pd.DataFrame()

    # Coerce numeric columns
    if 'lat' in df.columns and 'lon' in df.columns:
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')

        if df['lat'].abs().max() > 90:
            df['Long'] = df['lat']
            df['Lat'] = df['lon']
        else:
            df['Lat'] = df['lat']
            df['Long'] = df['lon']
    else:
        # Try common coordinate column names
        coord_candidates = [('Ycoord', 'Xcoord'), ('Y', 'X'), ('lon', 'lat')]
        found = False
        for a, b in coord_candidates:
            if a in df.columns and b in df.columns:
                df[a] = pd.to_numeric(df[a], errors='coerce')
                df[b] = pd.to_numeric(df[b], errors='coerce')
                df['Lat'] = df[a]
                df['Long'] = df[b]
                found = True
                break
        if not found:
            st.error('Không tìm thấy cột toạ độ (lat/lon hoặc Xcoord/Ycoord) trong customers.csv')
            return pd.DataFrame()

    # Ensure numeric ID
    df['ID'] = pd.to_numeric(df['ID'], errors='coerce').astype(int)

    # Drop rows with invalid coordinates
    df = df.dropna(subset=['Lat', 'Long'])
    return df

customers = load_customers("test90.csv")
if customers.empty:
    st.stop()

# Depot = ID=0
depot = customers.loc[customers['ID'] == 0].iloc[0]
depot_coord = (float(depot['Lat']), float(depot['Long']))

# --------------------------------
# ĐỌC FILE ROUTES
# --------------------------------
@st.cache_data
def load_routes():
    # Fallback: try solution JSON saved by solver
    json_p = _try_paths('results/solution_advanced.json')
    if json_p and json_p.exists():
        try:
            import json as _json
            with open(json_p, 'r', encoding='utf-8') as f:
                data = _json.load(f)
            # data may contain 'routes' as list of lists of customer IDs
            if 'routes' in data:
                return data['routes']
        except Exception:
            pass

    st.error("Không tìm thấy routes.txt hoặc results/solution_advanced.json để hiển thị tuyến.")
    return []

routes = load_routes()
if not routes:
    st.warning("Chưa có route nào để hiển thị.")
    st.stop()

# --------------------------------
# TÍNH CHI PHÍ THỰC TẾ BẰNG OSRM
# --------------------------------
@st.cache_data
def calculate_real_route_costs(routes, customers, depot_coord):
    """Tính chi phí thực tế cho tất cả các route bằng OSRM"""
    route_data = []
    total_real_cost = 0.0
    
    # Build mapping ID -> (lat, lon)
    latlon = {}
    for _, r in customers.iterrows():
        try:
            ident = int(r['ID'])
            lat = float(r['Lat'])
            lon = float(r['Long'])
            latlon[ident] = (lat, lon)
        except Exception:
            continue
    
    for idx, route_ids in enumerate(routes, start=1):
        # Build coordinate sequence: depot -> customers -> depot
        coords = [depot_coord]
        
        for rid in route_ids:
            if rid in latlon:
                coords.append(latlon[rid])
        
        coords.append(depot_coord)
        
        # Get real route from OSRM
        geom, real_distance_km = get_route_from_osrm(coords)
        
        # Calculate straight-line distance as fallback
        straight_distance = 0.0
        for i in range(len(coords) - 1):
            straight_distance += haversine_km(coords[i][0], coords[i][1], 
                                            coords[i+1][0], coords[i+1][1])
        
        # Use real distance if available, otherwise use straight-line
        if real_distance_km is not None:
            cost_km = real_distance_km
        else:
            cost_km = straight_distance
        
        route_data.append({
            "route_idx": idx,
            "route_ids": route_ids,
            "coordinates": coords,
            "geometry": geom,
            "real_distance_km": real_distance_km,
            "straight_distance_km": straight_distance,
            "final_distance_km": cost_km,
        })
        
        total_real_cost += cost_km
    
    return route_data, total_real_cost

# Tính toán chi phí thực tế
with st.spinner('Đang tính toán chi phí thực tế từ OSRM...'):
    route_data, total_real_cost = calculate_real_route_costs(routes, customers, depot_coord)

# --------------------------------
# HIỂN THỊ BẢNG ROUTE VỚI CHI PHÍ THỰC TẾ
# --------------------------------
st.subheader("Danh sách các tuyến xe với chi phí")

display_data = []
for data in route_data:
    display_data.append({
        "Route #": data["route_idx"],
        "Số điểm": len(data["route_ids"]),
        "Điểm đi qua": " → ".join(str(x) for x in data["route_ids"]),
        "Khoảng cách thực tế (km)": f"{data['final_distance_km']:.2f}" if data['real_distance_km'] is not None else f"{data['straight_distance_km']:.2f}*",
    })

df_routes = pd.DataFrame(display_data)
st.dataframe(df_routes, use_container_width=True, hide_index=True)

# Hiển thị tổng chi phí
col1, col2 = st.columns(2)
col1.metric("Tổng số tuyến", len(routes))
col2.metric("Tổng chi phí (km)", f"{total_real_cost:.2f}")

if any(data['real_distance_km'] is None for data in route_data):
    st.info("ℹMột số tuyến sử dụng khoảng cách ước tính (đường thẳng) do không thể kết nối đến OSRM")

# --------------------------------
# VẼ BẢN ĐỒ ROUTES VỚI ĐƯỜNG ĐI THỰC TẾ
# --------------------------------
st.subheader("Bản đồ các tuyến xe với đường đi")

# Tạo bản đồ
m = folium.Map(location=depot_coord, zoom_start=12)

# Vẽ depot (đỏ)
folium.Marker(
    depot_coord, 
    popup="Depot (ID=0)", 
    icon=folium.Icon(color="red", icon="home", prefix='fa')
).add_to(m)

# Vẽ các route với geometry thực tế từ OSRM
for data in route_data:
    color = random_color()
    route_idx = data["route_idx"]
    distance = data["final_distance_km"]
    
    # Sử dụng geometry thực tế nếu có, nếu không dùng đường thẳng
    if data["geometry"]:
        add_geojson_route_to_map(
            m, 
            data["geometry"], 
            color=color, 
            popup=f"Route #{route_idx} — {distance:.2f} km"
        )
    else:
        # Fallback: vẽ đường thẳng
        folium.PolyLine(
            data["coordinates"], 
            color=color, 
            weight=3, 
            opacity=0.9,
            popup=f"Route #{route_idx} — {distance:.2f} km"
        ).add_to(m)
    
    # Thêm markers cho các điểm khách hàng
    for seq, (lat, lon) in enumerate(data["coordinates"][1:-1], start=1):
        cust_id = data["route_ids"][seq-1] if (seq-1) < len(data["route_ids"]) else ''
        folium.CircleMarker(
            [lat, lon],
            radius=6,
            color=color,
            fill=True,
            fillOpacity=0.7,
            popup=f"Route {route_idx} - Điểm {seq} - ID {cust_id}",
        ).add_to(m)
    
    

# Hiển thị bản đồ
_ = st_folium(
    m,
    height=600,
    width=None,
    returned_objects=[],
)


st.markdown("---")
st.caption("Ứng dụng định tuyến xe - Hiển thị chi phí thực tế từ dữ liệu đường bộ")