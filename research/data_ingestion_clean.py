import os
import warnings
import zipfile
import gdown
from dataclasses import dataclass
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Change to project root
os.chdir("../")

from cnnClassifier.constants import *
from cnnClassifier.utils.common import *
from cnnClassifier import logger

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

class ConfigurationManager:
    def __init__(
        self, config_filepath = CONFIG_FILE_PATH, params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )
        return data_ingestion_config

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            
            print(f"Downloading file from: {dataset_url}")
            file_id = dataset_url.split('/')[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(f"{prefix}{file_id}", zip_download_dir, quiet=True)
            print(f"File downloaded successfully: {get_size(zip_download_dir)}")
            
        except Exception as e:
            raise e

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        print(f"Extracting zip file...")
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        print(f"Extraction completed. Files available at: {unzip_path}")

def main():
    try:
        print("Starting data ingestion process...")
        
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        
        print("Data ingestion completed successfully!")
        
    except Exception as e:
        print(f"Error during data ingestion: {e}")
        raise e

if __name__ == "__main__":
    main()
