

Env

    cd src
    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt


 
Docker

    docker pull osrm/osrm-backend
    docker run -t -v D:/VRPTW-Solver/app/data/raw:/data osrm/osrm-backend osrm-extract -p /opt/car.lua /data/vietnam-latest.osm.pbf
    docker run -t -v D:/VRPTW-Solver/app/data/raw:/data osrm/osrm-backend osrm-partition /data/vietnam-latest.osrm
    docker run -t -v D:/VRPTW-Solver/app/data/raw:/data osrm/osrm-backend osrm-customize /data/vietnam-latest.osrm
    docker run -d -p 5000:5000 -v D:/VRPTW-Solver/app/data/raw:/data osrm/osrm-backend osrm-routed --algorithm mld /data/vietnam-latest.osrm


streamlit run appication.py
