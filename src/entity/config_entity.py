import os
from dataclasses import dataclass, field
from datetime import datetime
from src.constants import *
from src.constants import DATA_VALIDATION_DIR_NAME 
from src.constants import DATA_TRANSFORMATION_DIR_NAME, TRANSFORM_OBJECT_FILE_NAME, TRANSFORMED_FILE_NAME,MODEL_CONFIG_FILE_PATH
import os
from src.constants import MODEL_EVALUATION_FILE_NAME

# Global timestamp
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

@dataclass
class DataIngestionConfig:
    training_pipeline_config: TrainingPipelineConfig = field(default_factory=TrainingPipelineConfig)
    data_ingestion_dir: str = field(init=False)
    feature_store_file_path: str = field(init=False)
    collection_name: str = field(default=DATA_INGESTION_COLLECTION_NAME)

    def __post_init__(self):
        self.data_ingestion_dir = os.path.join(
            self.training_pipeline_config.artifact_dir,
            DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_FEATURE_STORE_DIR,
            FILE_NAME
        )


@dataclass
class DataValidationConfig:
    training_pipeline_config: TrainingPipelineConfig
    data_validation_dir: str = None

    def __post_init__(self):
        self.data_validation_dir = os.path.join(
            self.training_pipeline_config.artifact_dir,
            DATA_VALIDATION_DIR_NAME
        )




@dataclass
class DataTransformationConfig:
    training_pipeline_config: TrainingPipelineConfig
    data_transformation_dir: str = field(init=False)
    transformer_object_path: str = field(init=False)
    transformed_data_path: str = field(init=False)

    def __post_init__(self):
        self.data_transformation_dir = os.path.join(
            self.training_pipeline_config.artifact_dir,
            DATA_TRANSFORMATION_DIR_NAME
        )
        self.transformer_object_path = os.path.join(
            self.data_transformation_dir,
            TRANSFORM_OBJECT_FILE_NAME
        )
        self.transformed_data_path = os.path.join(
            self.data_transformation_dir,
            TRANSFORMED_FILE_NAME
        )



@dataclass
class ModelTrainerConfig:
    training_pipeline_config: TrainingPipelineConfig = field(default_factory=TrainingPipelineConfig)
    trained_model_file_path: str = field(init=False)

    def __post_init__(self):
        self.trained_model_file_path = os.path.join(
            self.training_pipeline_config.artifact_dir,
            "model_trainer",
            "model.pkl"
        )


@dataclass
class ModelEvaluationConfig:
    training_pipeline_config: TrainingPipelineConfig
    model_evaluation_file_path: str = field(init=False)

    def __post_init__(self):
        self.model_evaluation_file_path = os.path.join(
            self.training_pipeline_config.artifact_dir,
            MODEL_EVALUATION_DIR_NAME,
            MODEL_EVALUATION_FILE_NAME
        )

