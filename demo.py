# main.py or app.py

from dotenv import load_dotenv
load_dotenv()

import dagshub
dagshub.init(
    repo_owner='shobanjatoth',
    repo_name='user-profile-Segmentation',
    mlflow=True
)

import mlflow
from src.pipline.training_pipeline import TrainPipeline  # Corrected spelling if needed

if __name__ == "__main__":
    with mlflow.start_run():
        obj = TrainPipeline()
        obj.run_pipeline()
