import os
import json
import pandas as pd
import requests

# === Cấu hình thư mục ===
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")

# ==============================
#   ĐỌC DỮ LIỆU TỪ CSV
# ==============================
def load_data(file_name):
    """
    Đọc file CSV chứa NUM_VEHICLE, CAPACITY và dữ liệu khách hàng + depot.
    Trả về:
        - vehicle_number (int)
        - capacity (float)
        - df (DataFrame chứa dữ liệu depot và khách hàng)
        - locations (list of tuples (lat, lon))
    """
    path = os.path.join(DATA_RAW, file_name)
    
    # Đọc hai dòng đầu tiên để lấy NUM_VEHICLE và CAPACITY
    with open(path, 'r') as f:
        lines = f.readlines()
        vehicle_number = int(lines[0].strip().split('=')[1])  # NUM_VEHICLE
        capacity = float(lines[1].strip().split('=')[1])      # CAPACITY
    
    # Đọc dữ liệu khách hàng và depot từ dòng 3 (bỏ qua 2 dòng đầu)
    df = pd.read_csv(path, skiprows=2)
    
    # Tọa độ (LAT, LON)
    locations = list(zip(df["lat"], df["lon"]))
    
    return vehicle_number, capacity, df, locations

# ==============================
#   TÍNH MA TRẬN THỜI GIAN BẰNG OSRM LOCAL
# ==============================
def compute_time_matrix(locations):
    """
    Tính ma trận thời gian (giây) giữa các điểm.
    Sử dụng khoảng cách Euclidean với hệ số chuyển đổi thời gian.
    locations: list of (lat, lon)
    osrm_url: URL của OSRM server (không sử dụng để tránh lỗi)
    Trả về: matrix n x n
    """
    import math
    
    n = len(locations)
    print(f"Tính ma trận  {n} điểm")
    
    # Khởi tạo ma trận kết quả
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    
    # Hệ số chuyển đổi từ khoảng cách (độ) sang thời gian (giây)
    # Giả sử tốc độ trung bình 30 km/h = 8.33 m/s
    # 1 độ ≈ 111 km, vậy 1 độ ≈ 111000 m
    # Thời gian = khoảng cách / tốc độ = (độ * 111000) / 8.33
    DEGREE_TO_SECONDS = 111000 / 8.33  # ≈ 13320 giây/độ
    
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0.0
            else:
                # Tính khoảng cách Euclidean
                lat1, lon1 = locations[i]
                lat2, lon2 = locations[j]
                distance = math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)
                
                # Chuyển đổi sang thời gian (giây)
                matrix[i][j] = distance * DEGREE_TO_SECONDS
    
    print(f"Hoàn thành ma trận {len(matrix)}x{len(matrix[0])}")
    return matrix

def compute_time_matrix_OSRM(locations, osrm_url="http://localhost:5000"):
    """
    Tính ma trận thời gian (giây) giữa các điểm bằng OSRM local.
    locations: list of (lat, lon)
    osrm_url: URL của OSRM server
    Trả về: matrix n x n
    """

    # OSRM expects coordinates in the format "longitude,latitude"
    # But our data has them swapped in the CSV, so we use them as is
    coords = ";".join([f"{lat},{lon}" for lat, lon in locations])
    url = f"{osrm_url}/table/v1/driving/{coords}"
    params = {"annotations": "duration"}  # chỉ lấy thời gian, không cần khoảng cách

    print(f"Gọi OSRM local ({len(locations)} điểm)...")
    r = requests.get(url, params=params)
    if r.status_code != 200:
        raise Exception(f"Lỗi khi gọi OSRM: {r.status_code} - {r.text}")

    data = r.json()
    matrix = data.get("durations", [])
    print(f"Nhận ma trận kích thước {len(matrix)}x{len(matrix[0])}")
    return matrix

# ==============================
#   LƯU MA TRẬN
# ==============================
def save_matrix(matrix, filename="time_matrix_osrm.json", to_minutes: bool = True):
    """
    Lưu ma trận thời gian vào file JSON.

    Nếu to_minutes=True, chuyển các giá trị thời gian (giây) sang phút trước khi lưu.
    Giá trị không hợp lệ (None / non-finite / inf) được lưu dưới dạng null (JSON null) để dễ xử lý khi load.
    """
    import math

    path = os.path.join(DATA_PROCESSED, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Chuẩn hóa và (tuỳ chọn) chuyển sang phút
    out = []
    for row in matrix:
        prow = []
        for v in row:
            try:
                if v is None:
                    prow.append(None)
                else:
                    # đảm bảo là số
                    fv = float(v)
                    if not math.isfinite(fv):
                        prow.append(None)
                    else:
                        if to_minutes:
                            prow.append(fv / 60.0)
                        else:
                            prow.append(fv)
            except Exception:
                prow.append(None)
        out.append(prow)

    with open(path, "w") as f:
        json.dump(out, f)
    print(f"Ma trận đã lưu vào: {path} (to_minutes={to_minutes})")

# ==============================
#   CHUYỂN ĐỊNH DẠNG THỜI GIAN
# ==============================
def time_to_minutes(t):
    """Chuyển HH:MM thành phút kể từ 00:00."""
    if isinstance(t, (int, float)):
        return int(t)
    if isinstance(t, str):
        h, m = map(int, t.split(":"))
        return h * 60 + m
    return 0

# ==============================
#   CHUẨN BỊ DỮ LIỆU VRPTW
# ==============================
def load_data_with_tw(file_name):
    """
    Chuẩn bị dữ liệu VRPTW từ file CSV với định dạng:
    NUM_VEHICLE=<value>
    CAPACITY=<value>
    ID,LAT,LON,DEMAND,READY_TIME,DUE_DATE,SERVICE_TIME
    Trả về:
        - data dict (vehicle_number, capacity, depot, customers, num_customers, id_to_idx, idx_to_id, customer_map)
        - locations (list of tuples (lat, lon))
    """
    # Đọc file CSV
    vehicle_number, capacity, df, locations = load_data(file_name)
    
    # Chuyển đổi thời gian sang phút
    for col in ["READY_TIME", "DUE_DATE", "SERVICE_TIME"]:
        if col in df.columns:
            df[col] = df[col].apply(time_to_minutes)

    # Tách depot (ID=0) và khách hàng
    depot_df = df[df["ID"] == 0]
    customers_df = df[df["ID"] != 0]

    if depot_df.empty:
        raise ValueError("Không tìm thấy depot (ID=0) trong file dữ liệu")

    # Chuyển đổi sang danh sách dictionary
    depot = depot_df.to_dict(orient="records")[0]
    customers = customers_df.to_dict(orient="records")

    # Tạo ánh xạ ID và chỉ số
    id_to_idx = {0: 0}  # Depot luôn có chỉ số 0
    idx_to_id = {0: 0}
    customer_map = {0: depot}  # Lưu thông tin depot

    for idx, cust in enumerate(customers, 1):  # Bắt đầu từ 1 vì 0 là depot
        id_to_idx[cust["ID"]] = idx
        idx_to_id[idx] = cust["ID"]
        customer_map[idx] = cust

    # Depot với chỉ số idx
    depot_data = {
        "idx": 0,
        "orig_id": depot["ID"],
        "x": depot["lat"],
        "y": depot["lon"],
        "demand": depot["DEMAND"],
        "ready_time": depot["READY_TIME"],
        "due_time": depot["DUE_DATE"],
        "service_time": depot["SERVICE_TIME"]
    }
    
    # Customers với chỉ số idx
    customers_data = []
    for idx, cust in enumerate(customers, 1):
        cust_data = {
            "idx": idx,
            "orig_id": cust["ID"],
            "x": cust["lat"],
            "y": cust["lon"],
            "demand": cust["DEMAND"],
            "ready_time": cust["READY_TIME"],
            "due_time": cust["DUE_DATE"],
            "service_time": cust["SERVICE_TIME"],
            "tw_length": cust["DUE_DATE"] - cust["READY_TIME"]
        }
        customers_data.append(cust_data)
        customer_map[idx] = cust_data
    
    # Cập nhật depot trong customer_map
    customer_map[0] = depot_data
    
    # Tạo từ điển data
    data = {
        "vehicle_number": vehicle_number,
        "capacity": capacity,
        "depot": depot_data,
        "customers": customers_data,
        "num_customers": len(customers_data),
        "id_to_idx": id_to_idx,
        "idx_to_id": idx_to_id,
        "customer_map": customer_map
    }

    # Kiểm tra tính khả thi của tổng nhu cầu
    total_demand = sum(cust["DEMAND"] for cust in customers)
    if total_demand > vehicle_number * capacity:
        print(f"Tổng nhu cầu ({total_demand}) vượt quá dung lượng đội xe ({vehicle_number * capacity})")

    return data, locations