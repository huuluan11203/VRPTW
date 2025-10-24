# pages/02_quan_ly_cache_tuyen.py
from pathlib import Path
import os, json, math, datetime
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Quản lý cache tuyến (route_geoms)", page_icon="🧭", layout="wide")
st.markdown(
    "<h2 style='text-align:center;margin:0.25rem 0 1rem 0'>Quản lý cache tuyến (route_geoms)</h2>",
    unsafe_allow_html=True,
)

# giữ key các tuyến cần hiển thị sau khi nhấn "Xem"
if "cache_view_keys" not in st.session_state:
    st.session_state.cache_view_keys = []


APP_DIR = Path(__file__).resolve().parents[1]
GEOMS_DIR = APP_DIR / "results" / "route_geoms"
GEOMS_DIR.mkdir(parents=True, exist_ok=True)

st.info(f"Thư mục hiện tại: `{GEOMS_DIR}`")

def _list_cache_pairs(folder: Path):
    """
    Gom theo 'key' (tên file không gồm đuôi). Mỗi key có thể có .geojson và/hoặc .distance.
    Trả về DataFrame: key, has_geojson, geojson_size, geojson_mtime, has_distance, distance_size, distance_mtime.
    """
    files = list(folder.glob("*"))
    meta = {}
    for p in files:
        stem = p.stem  # tên không gồm đuôi
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
        return pd.DataFrame(columns=[
            "key","has_geojson","geojson_size","geojson_mtime","has_distance","distance_size","distance_mtime"
        ])
    df = pd.DataFrame(list(meta.values()))
    # sắp xếp mới nhất trước (ưu tiên theo geojson/distance mtime lớn nhất)
    df["last_mtime"] = df[["geojson_mtime","distance_mtime"]].max(axis=1)
    df = df.sort_values(by="last_mtime", ascending=False).reset_index(drop=True)
    return df

def _human_size(n):
    if n is None or n == 0: return "0 B"
    units = ["B","KB","MB","GB","TB"]
    i = int(math.floor(math.log(max(n,1), 1024)))
    return f"{n/1024**i:.1f} {units[i]}"

def _read_distance_km(folder: Path, key: str):
    """Đọc giá trị km từ file <key>.distance. Trả về float hoặc None."""
    p = folder / f"{key}.distance"
    try:
        return float(p.read_text(encoding="utf-8").strip())
    except Exception:
        return None


df = _list_cache_pairs(GEOMS_DIR)

# ===== Bộ lọc & lựa chọn =====
left, right = st.columns([2,3])
with left:
    st.markdown("#### Danh sách cache")
    if df.empty:
        st.warning("Chưa có file cache trong thư mục.")
    else:
        df_show = df.copy()
        df_show["geojson_size"] = df_show["geojson_size"].map(_human_size)
        df_show["distance_size"] = df_show["distance_size"].map(_human_size)
        df_show = df_show.rename(columns={
            "key":"Key",
            "has_geojson":"GeoJSON?",
            "geojson_size":"GeoJSON Size",
            "geojson_mtime":"GeoJSON Modified",
            "has_distance":"Distance?",
            "distance_size":"Distance Size",
            "distance_mtime":"Distance Modified",
            "last_mtime":"Last Modified",
            "distance_km": "Distance (km)",

        })
        
        # đọc distance_km cho từng key (nếu có .distance)
        df["distance_km"] = [
            _read_distance_km(GEOMS_DIR, k) if has else None
            for k, has in zip(df["key"], df["has_distance"])
        ]

        df_show["distance_km"] = df["distance_km"].map(lambda x: None if pd.isna(x) else round(float(x), 2))



        st.dataframe(df_show, use_container_width=True, hide_index=True)

        keys = df["key"].tolist()
        pick = st.multiselect("Chọn key để xem/xoá:", keys, max_selections=10)

        c1, c2, c3 = st.columns([1,1,2])
        view_btn = c1.button("Xem trên bản đồ", use_container_width=True, disabled=not pick)
        del_btn  = c2.button("Xoá các mục đã chọn", use_container_width=True, disabled=not pick)
        clear_btn = c3.button("Xoá toàn bộ cache", use_container_width=True, disabled=df.empty)

        # Khi bấm Xem → lưu lựa chọn vào state
        if view_btn and pick:
            st.session_state.cache_view_keys = list(pick)


with right:
    st.markdown("#### Bản đồ xem trước")
    m = folium.Map(location=[21.0285,105.8542], zoom_start=12)

    # Nếu bấm xem → vẽ các tuyến theo GeoJSON
    if df.empty:
        st.info("Không có tuyến nào để xem.")
    else:
        # to_draw = pick if view_btn and pick else []

        # Luôn vẽ theo state đã lưu (giữ được sau rerun)
        to_draw = st.session_state.get("cache_view_keys", [])
        # Nếu có key đã bị xoá khỏi thư mục, loại ra:
        existing = {p.stem for p in GEOMS_DIR.glob("*.geojson")}
        to_draw = [k for k in to_draw if (GEOMS_DIR / f"{k}.geojson").exists()]
        # (tuỳ chọn) nếu muốn tự động cập nhật state sau khi lọc:
        st.session_state.cache_view_keys = to_draw

        if to_draw:
            pal = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                   "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
            for i, k in enumerate(to_draw):
                gj = GEOMS_DIR / f"{k}.geojson"
                color = pal[i % len(pal)]
                if gj.exists():
                    try:
                        geom = json.loads(gj.read_text(encoding="utf-8"))
                        gj_layer = folium.GeoJson(
                        data={"type": "Feature", "geometry": geom},
                        style_function=lambda feat, col=color: {"color": col, "weight": 4, "opacity": 0.85}
                        ).add_to(m)

                        # thêm popup tại điểm giữa tuyến
                        coords = geom.get("coordinates", [])
                        if coords:
                            mid = coords[len(coords)//2]  # [lon, lat]
                            dkm = _read_distance_km(GEOMS_DIR, k)
                            folium.Marker(
                                [mid[1], mid[0]],
                                popup=f"{k} — {dkm:.2f} km" if dkm is not None else f"{k} — (chưa có .distance)"
                            ).add_to(m)

                    except Exception as e:
                        st.error(f"Lỗi đọc {gj.name}: {e}")
                else:
                    st.warning(f"Không có GeoJSON cho key {k}.")
        else:
            st.caption("Chọn các key ở khung bên trái rồi nhấn **👀 Xem trên bản đồ** để hiển thị.")

    st_folium(m, height=620, use_container_width=True, key="map_cache_preview")

    # Bảng tóm tắt distance cho các tuyến đang được hiển thị
if to_draw:
    sel_dist = []
    total_km = 0.0
    for k in to_draw:
        dkm = _read_distance_km(GEOMS_DIR, k)
        if dkm is not None:
            total_km += dkm
        sel_dist.append({"Key": k, "Distance (km)": None if dkm is None else round(dkm, 2)})
    st.markdown("##### Khoảng cách các tuyến đang xem")
    st.dataframe(pd.DataFrame(sel_dist), use_container_width=True, hide_index=True)
    st.metric("Tổng quãng đường (km)", f"{total_km:.2f}")


# ===== Xử lý xoá =====
if del_btn and pick:
    ok = 0
    for k in pick:
        for ext in (".geojson",".distance"):
            p = GEOMS_DIR / f"{k}{ext}"
            if p.exists():
                try:
                    p.unlink()
                    ok += 1
                except Exception as e:
                    st.error(f"Không xoá được {p.name}: {e}")
    st.success(f"Đã xoá {ok} file.")
    st.rerun()

if clear_btn and not df.empty:
    ok = 0
    for p in GEOMS_DIR.glob("*"):
        try:
            p.unlink(); ok += 1
        except Exception as e:
            st.error(f"Không xoá được {p.name}: {e}")
    st.success(f"Đã xoá {ok} file trong route_geoms.")
    st.rerun()

st.markdown(
    "<div style='text-align:center;margin-top:1rem'>"
    "<a href='/' target='_self' style='text-decoration:none'>⬅️ Quay lại trang chính</a>"
    "</div>",
    unsafe_allow_html=True,
)
