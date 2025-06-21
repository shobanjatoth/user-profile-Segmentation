import os
import sys
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

from src.exception import USvisaException
from src.logger import logging
from src.utils.main_utils import (
    load_object,
    read_yaml_file,
    write_yaml_file,
    save_object  # ✅ Make sure this exists
)

from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelEvaluationArtifact


class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("🚀 Starting model evaluation")

            # Load transformed data
            transformed_data = pd.read_csv(self.data_transformation_artifact.transformed_data_path)

            # Load model config
            model_config = read_yaml_file(os.path.join("config", "model.yaml"))
            model_params = model_config["model_selection"]["module_0"]["params"]

            # Fit new model
            new_model = KMeans(**model_params)
            new_model.fit(transformed_data)

            new_score = silhouette_score(transformed_data, new_model.labels_)

            # Save evaluation report
            eval_report = {
                "model_evaluation": {
                    "silhouette_score": float(new_score)
                }
            }

            os.makedirs(os.path.dirname(self.model_eval_config.model_evaluation_file_path), exist_ok=True)
            write_yaml_file(self.model_eval_config.model_evaluation_file_path, content=eval_report)

            # Save evaluated model
            evaluated_model_path = os.path.join(
                os.path.dirname(self.model_eval_config.model_evaluation_file_path),
                "evaluated_model.pkl"
            )
            save_object(evaluated_model_path, new_model)

            logging.info(f"✅ Model evaluation completed. Silhouette Score: {new_score:.4f}")

            return ModelEvaluationArtifact(
                is_model_accepted=True,
                evaluated_model_path=evaluated_model_path,
                silhouette_score=new_score,
                improved_accuracy=new_score  # You can subtract previous_score if comparing
            )

        except Exception as e:
            raise USvisaException(e, sys)

