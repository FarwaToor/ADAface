#!/usr/bin/env python3
import gdown



def download_yolov8_model(model_url, output_path):

    gdown.download(model_url, output_path, quiet=False)



if __name__ == "__main__":

    model_url = 'https://drive.google.com/uc?id=1IJZBcyMHGhzAi0G4aZLcqryqZSjPsps-'

    output_path = 'yolov8m_200e.pt'

    download_yolov8_model(model_url, output_path)

