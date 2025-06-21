import os
from datetime import date

DATABASE_NAME = "user-profile"
COLLECTION_NAME = "segmentation"

MONGODB_URL = "mongodb+srv://217r1a67f1:bWTGsEoGuuZwxoS7@nayak.movm0en.mongodb.net/?retryWrites=true&w=majority&appName=nayak"

PIPELINE_NAME: str = "src"
ARTIFACT_DIR: str = "artifact"

FILE_NAME: str = "user-profilesegmentation.csv"

SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
MODEL_CONFIG_FILE_PATH = os.path.join("config", "model.yaml")

# Data Ingestion Constants
DATA_INGESTION_COLLECTION_NAME: str = COLLECTION_NAME  # corrected
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"

# Data vallidation

DATA_VALIDATION_DIR_NAME = "data_validation"

# Data Transformation Constants
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
TRANSFORMED_DATA_DIR: str = "transformed"
TRANSFORM_OBJECT_FILE_NAME: str = "transformer.pkl"
TRANSFORMED_FILE_NAME: str = "transformed_data.csv"

# Model Trainer
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_OBJECT_FILE_NAME: str = "model.pkl"
MODEL_TRAINER_SAVED_MODEL_DIR: str = "saved_models"

# Model Evaluation
# -------------------------------
MODEL_EVALUATION_DIR_NAME = "model_evaluation"
MODEL_EVALUATION_FILE_NAME = "model_evaluation.yaml"



