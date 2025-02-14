from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import torch
from src.model import TornadoDetectionModel, bilinear_interpolation
from src.data_processor import RadarDataProcessor
from src.data_downloader import RadarDataDownloader
import os
import torch.nn.functional as F
from src.constant import RADARS_LOCATION
import numpy as np
from src.utils import get_points_at_distance
app = FastAPI(title="Tornado Detection API")


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to restrict access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_processed_data(data_downloader, data_processor, date, timestamp):
    try:
        lv2_radar_file = data_downloader.download_nextrad_2_file(date, timestamp)
        lv2_radar_data = data_downloader.read_nextrad_2_file(lv2_radar_file)
        # os.remove(lv2_radar_file)
        kdp_radar_data_tilt05 = data_downloader.get_nexrad_l3_data(product='N0K', date=date, time=timestamp)
        kdp_radar_data_tilt09 = data_downloader.get_nexrad_l3_data(product='N3K', date=date, time=timestamp)
        preprocessed_data = data_processor.preprocess(lv2_radar_data, kdp_radar_data_tilt05, kdp_radar_data_tilt09)
        return preprocessed_data
    except Exception as e:
        return None

model = None
data_downloaders = None
data_processors = None

@app.on_event("startup")
async def startup_event():
    global model
    model = TornadoDetectionModel(
        model_path="model_checkpoint/tornet.pth",
        num_range=1152,
        include_range_folded=True
    )
    global data_processors
    data_processors = RadarDataProcessor(num_range=1152)

    global data_downloaders
    data_downloaders = []

class RadarData(BaseModel):
    radar_station: str
    date: str
    current_timestamp: str

@app.post("/predict")
async def predict(radar_data: RadarData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    data_downloader = [x for x in data_downloaders if x.radar_station == radar_data.radar_station]
    if len(data_downloader) == 0:
        data_downloader = RadarDataDownloader(
            radar_station=radar_data.radar_station,
            dest_folder='data/lv2_data'
        )
        data_downloaders.append(data_downloader)
    data_downloader = data_downloaders[-1]
    list_available_time = data_downloader.get_available_time(radar_data.date)
    last_updated_timestamp = list_available_time[-1]
    #if last updated timestamp is newer than current timestamp, then download the file
    #the timestamp is in the format of HHMMSS
    # TODO: check corner case when the date changes.
    if last_updated_timestamp > radar_data.current_timestamp:
        preprocessed_data = get_processed_data(data_downloader, data_processors, radar_data.date, last_updated_timestamp)
        if preprocessed_data is None:
            return {"probability": "-1"}
        return {"probability": model.predict(preprocessed_data)}
    else:
        return {"probability": "-1"}

class TornadoPoint(BaseModel):
    pinpoint: tuple[float, float]  # (lat, lon) tuple
    probability: float  # Probability score for this detection 
    size: int  # Size in pixels of detected region
    cropped_dbz: List[List[float]]  # Cropped DBZ data around detection
    cropped_vel: List[List[float]]  # Cropped velocity data around detection

class TornadoDetection(BaseModel):
    detections: List[TornadoPoint]

@app.post("/detect_tornado", response_model=Union[TornadoDetection, dict])
async def detect_tornado(radar_data: RadarData):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        data_downloader = [x for x in data_downloaders if x.radar_station == radar_data.radar_station]
        if len(data_downloader) == 0:
            data_downloader = RadarDataDownloader(
                radar_station=radar_data.radar_station,
                dest_folder='data/lv2_data'
            )
            data_downloaders.append(data_downloader)
        data_downloader = data_downloaders[-1]
        list_available_time = data_downloader.get_available_time(radar_data.date)
        last_updated_timestamp = list_available_time[-1]
        #if last updated timestamp is newer than current timestamp, then download the file
        #the timestamp is in the format of HHMMSS
        # TODO: check corner case when the date changes.
        if last_updated_timestamp > radar_data.current_timestamp:
            preprocessed_data = get_processed_data(data_downloader, data_processors, radar_data.date, last_updated_timestamp)
            if preprocessed_data is None:
                return {"error": "Failed to process radar data", "status": "error"}
            likelihood = model.calculate_likelihood(preprocessed_data)
            likelihood = bilinear_interpolation(likelihood.squeeze(), (720, model.num_range))
            probability_map = torch.sigmoid(likelihood)
            detected_centroids, detected_probabilities, detected_sizes = model.detect_tornados(probability_map.numpy())

            #get the 120*240 pixel around the detected centroids
            DBZ_data = preprocessed_data['DBZ'][..., 0]
            DBZ_data = DBZ_data.squeeze()
            VEL_data = preprocessed_data['VEL'][..., 0]
            VEL_data = VEL_data.squeeze()

            cropped_DBZ_data = []
            cropped_VEL_data = []
            for centroid in detected_centroids:
                x, y = centroid
                x_min = max(0, x - 60)
                x_max = min(719, x + 60)
                y_min = max(0, y - 120)
                y_max = min(model.num_range, y + 120)
                cropped_DBZ_data.append(DBZ_data[x_min:x_max, y_min:y_max])
                cropped_VEL_data.append(VEL_data[x_min:x_max, y_min:y_max])

            #and get the lat and lon of the centroids
            radar_lat, radar_lon = RADARS_LOCATION[radar_data.radar_station]
            tornado_points = []
            for i, centroid in enumerate(detected_centroids):
                x, y = centroid
                range_km = x * 0.25 + 0.25
                azimuth = np.degrees(y * (2*np.pi/720))
                lat, lon = get_points_at_distance(radar_lat, radar_lon, range_km, azimuth)
                tornado_points.append(TornadoPoint(
                    pinpoint=(lat, lon),
                    probability=detected_probabilities[i],
                    size=detected_sizes[i],
                    cropped_dbz=cropped_DBZ_data[i].tolist(),
                    cropped_vel=cropped_VEL_data[i].tolist()
                ))

            return TornadoDetection(detections=tornado_points)
        else:
            return {"error": "No new data available", "status": "no_update"}
    except Exception as e:
        return {"error": str(e), "status": "error"}

if __name__ == "__main__":
    # Initialize model and processor
    model = TornadoDetectionModel(
        model_path="model_checkpoint/tornet.pth",
        num_range=1152,
        include_range_folded=True
    )
    data_processors = RadarDataProcessor(num_range=1152)
    
    # Test data
    test_radar_data = RadarData(
        radar_station="KDVN",
        date="20240314",
        current_timestamp="000610"
    )
    
    # Create data downloader
    data_downloader = RadarDataDownloader(
        radar_station=test_radar_data.radar_station,
        dest_folder='data/lv2_data'
    )
    
    # Get test prediction
    list_available_time = data_downloader.get_available_time(test_radar_data.date)
    # print(list_available_time)
    last_updated_timestamp = list_available_time[-1]
    
    if last_updated_timestamp > test_radar_data.current_timestamp:
        preprocessed_data = get_processed_data(data_downloader, data_processors, test_radar_data.date, "093012")
        if preprocessed_data is not None:
            likelihood = model.calculate_likelihood(preprocessed_data)
            probability = model.predict(preprocessed_data)
            likelihood = bilinear_interpolation(likelihood.squeeze(), (720, model.num_range))
            probability_map = torch.sigmoid(likelihood)
            print(f"Test prediction probability: {probability}")
            detected_centroids, detected_probabilities, detected_sizes = model.detect_tornados(probability_map.numpy())
            
            radar_lat, radar_lon = RADARS_LOCATION[test_radar_data.radar_station]
            lat_lon_centroids = []
            for centroid in detected_centroids:
                x, y = centroid
                # Convert from grid coordinates to range/azimuth
                range_km = x * 0.25 + 0.25
                azimuth = np.degrees(y * (2*np.pi/720))
                lat, lon = get_points_at_distance(radar_lat, radar_lon, range_km, azimuth)
                lat_lon_centroids.append((lat, lon))
            
            DBZ_data = preprocessed_data['DBZ'][..., 0]
            DBZ_data = DBZ_data.squeeze()
            print(DBZ_data.shape)
            VEL_data = preprocessed_data['VEL'][..., 0]
            VEL_data = VEL_data.squeeze()

            cropped_DBZ_data = []
            cropped_VEL_data = []
            for centroid in detected_centroids:
                x, y = centroid
                x_min = max(0, int(x - 60))
                x_max = min(719, int(x + 60))
                y_min = max(0, int(y - 120))
                y_max = min(model.num_range, int(y + 120))
                cropped_DBZ_data.append(DBZ_data[x_min:x_max, y_min:y_max])
                cropped_VEL_data.append(VEL_data[x_min:x_max, y_min:y_max])
            
            print(f"Test prediction detected_centroids: {detected_centroids}")
            print(f"Test prediction detected_probabilities: {detected_probabilities}")
            print(f"Test prediction detected_sizes: {detected_sizes}")
            print(f"Test prediction lat_lon_centroids: {lat_lon_centroids}")
            print(f"Test prediction cropped_DBZ_data: {cropped_DBZ_data[0].shape}")
            print(f"Test prediction cropped_VEL_data: {cropped_VEL_data[0].shape}")
        else:
            print("Failed to get preprocessed data")
    else:
        print("No newer data available")