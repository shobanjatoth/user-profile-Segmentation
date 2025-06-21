import os
import sys
import pandas as pd
import importlib
from sklearn.metrics import silhouette_score
from src.logger import logging
from src.exception import USvisaException
from src.utils.main_utils import read_yaml_file, save_object
from src.constants import MODEL_CONFIG_FILE_PATH
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

# ✅ MLflow & .env setup
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")


class ModelTrainer:
    def __init__(self, 
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            self.model_config = read_yaml_file(MODEL_CONFIG_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    def train_model(self) -> ModelTrainerArtifact:
        try:
            logging.info("🚀 Loading transformed data")
            df = pd.read_csv(self.data_transformation_artifact.transformed_data_path)

            best_model = None
            best_score = -1
            best_model_name = None

            # ✅ Start MLflow experiment
            mlflow.set_experiment("Model_Trainer_Clustering")

            with mlflow.start_run(run_name="ClusterModelSearch"):
                mlflow.log_param("transformed_data_path", self.data_transformation_artifact.transformed_data_path)

                logging.info("🔍 Searching best model from model.yaml")
                for model_key, model_info in self.model_config['model_selection'].items():
                    class_name = model_info['class']
                    module_name = model_info['module']
                    params = model_info['params']

                    model_class = getattr(importlib.import_module(module_name), class_name)
                    model = model_class(**params)

                    model.fit(df)
                    clusters = model.predict(df)
                    score = silhouette_score(df, clusters)

                    logging.info(f"Model: {class_name}, Silhouette Score: {score}")

                    # ✅ Log each model's params and metric
                    mlflow.log_param(f"{class_name}_params", str(params))
                    mlflow.log_metric(f"{class_name}_silhouette", score)

                    if score > best_score:
                        best_model = model
                        best_score = score
                        best_model_name = class_name

                if best_model is None:
                    raise Exception("❌ No model was successfully trained.")

                os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
                save_object(self.model_trainer_config.trained_model_file_path, best_model)

                logging.info(f"✅ Best Model: {best_model_name}, Score: {best_score}")

                # ✅ Log best model details and artifact
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_metric("best_silhouette_score", best_score)
                mlflow.sklearn.log_model(best_model, artifact_path="best_model")

                return ModelTrainerArtifact(
                    model_path=self.model_trainer_config.trained_model_file_path,
                    silhouette_score=best_score
                )

        except Exception as e:
            raise USvisaException(e, sys)



