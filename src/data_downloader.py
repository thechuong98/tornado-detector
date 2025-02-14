import boto3
from botocore import UNSIGNED
from botocore.config import Config
import os
import pyart

class RadarDataDownloader:
    def __init__(self, radar_station: str, dest_folder: str):
        self.radar_station = radar_station
        self.dest_folder = dest_folder
        self.s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    def get_available_time(self, date, hour=None):
        """
        Get a list of all available times for a given station and date.
        
        Parameters:
        -----------
        station : str
            Four-letter radar station identifier
        date : str
            Date in YYYYMMDD format
        """
        assert self.radar_station is not None, "Radar station is not set"
        assert len(date) == 8, "Date must be in YYYYMMDD format"
        bucket = 'noaa-nexrad-level2'
        
        # Construct the prefix
        year = date[:4]
        month = date[4:6]
        day = date[6:8]
        
        if hour is not None:
            prefix = f"{year}/{month}/{day}/{self.radar_station}/{self.radar_station}{date}_{hour}"
        else:
            prefix = f"{year}/{month}/{day}/{self.radar_station}/{self.radar_station}{date}"
        
        # List objects with the prefix
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            files = []
            
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                if 'Contents' in page:
                    files.extend([obj['Key'] for obj in page['Contents']])
                    
        except Exception as e:
            print(f"Error listing files: {str(e)}")
            files = []

        times = []
        for f in files:
            try:
                time = f.split('_')[1]
                times.append(time)
            except:
                continue
        
        times.sort()
        return times

    def download_nextrad_2_file(self, date, time):
        year = date[:4]
        month = date[4:6]
        day = date[6:8]
        key = f"{year}/{month}/{day}/{self.radar_station}/{self.radar_station}{date}_{time}_V06"
        bucket = 'noaa-nexrad-level2'
        os.makedirs(self.dest_folder, exist_ok=True)
        local_file = f"{self.dest_folder}/{self.radar_station}{date}_{time}_V06"
        if os.path.exists(local_file):
            return local_file
        try:
            self.s3.download_file(bucket, key, local_file)
            return local_file
        except Exception as e:
            print(f"Error downloading file: {str(e)}")
            return None
        
    def read_nextrad_2_file(self, file_path):
        if os.path.exists(file_path):
            try:
                radar_lv2_data = pyart.io.read_nexrad_archive(file_path)
                return radar_lv2_data
            except Exception as e:
                print(f"Error reading file: {str(e)}")
                return None
        else:
            print(f"File {file_path} does not exist")
            return None
        
    
    # def read_nextrad_2_file(self, year, month, day, time):
    #     s3_link = f"s3://noaa-nexrad-level2/{year}/{month}/{day}/{self.radar_station}/{self.radar_station}{year}{month}{day}_{time}_V06"
    #     radar_lv2_data = pyart.io.read_nexrad_archive(s3_link)
    #     return radar_lv2_data
    

    def get_nexrad_l3_data(self, product: str, date: str, time: str):
        if len(self.radar_station) == 4: # KTLX -> TLX
            radar_station = self.radar_station[1:]
        else:
            radar_station = self.radar_station
        year = date[:4]
        month = date[4:6]
        day = date[6:8]
        hour = time[:2]
        minute = time[2:4]
        second = time[4:6]
        s3_link = f"s3://unidata-nexrad-level3/{radar_station}_{product}_{year}_{month}_{day}_{hour}_{minute}_{second}"
        radar_lv3_data = pyart.io.read_nexrad_level3(s3_link)
        return radar_lv3_data

if __name__ == "__main__":
    downloader = RadarDataDownloader(radar_station="KTLX", dest_folder="./data")
    print(downloader.get_available_time(date="20250212"))
