from dataclasses import dataclass
from typing import Any

@dataclass
class DataIngestionArtifact:
    feature_store_file_path: str



@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str



@dataclass
class DataTransformationArtifact:
    transformed_data_path: str
    transformer_object_path: str



@dataclass
class ModelTrainerArtifact:
    model_path: str
    silhouette_score: float




@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    evaluated_model_path: str
    silhouette_score: float
    improved_accuracy: float

