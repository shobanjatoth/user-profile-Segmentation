from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    TrainingPipelineConfig
)

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ModelEvaluationArtifact
)

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

from src.logger import logging
from src.exception import USvisaException
import sys


class TrainPipeline:
    def __init__(self):
        try:
            logging.info("🚀 Initializing TrainPipeline")
            self.training_pipeline_config = TrainingPipelineConfig()

            self.data_ingestion_config = DataIngestionConfig(self.training_pipeline_config)
            self.data_validation_config = DataValidationConfig(self.training_pipeline_config)
            self.data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
            self.model_trainer_config = ModelTrainerConfig()
            self.model_evaluation_config = ModelEvaluationConfig(self.training_pipeline_config)

        except Exception as e:
            raise USvisaException(e, sys)

    def start_data_ingestion(self) -> DataIngestionArtifact:
        logging.info("📥 Starting data ingestion")
        ingestion = DataIngestion(self.data_ingestion_config)
        return ingestion.initiate_data_ingestion()

    def start_data_validation(self, ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        logging.info("🔍 Starting data validation")
        validation = DataValidation(ingestion_artifact, self.data_validation_config)
        return validation.initiate_data_validation()

    def start_data_transformation(self, ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        logging.info("🔧 Starting data transformation")
        transformation = DataTransformation(ingestion_artifact, self.data_transformation_config)
        return transformation.initiate_data_transformation()

    def start_model_trainer(self, transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        logging.info("🤖 Starting model training")
        trainer = ModelTrainer(transformation_artifact, self.model_trainer_config)
        return trainer.train_model()

    def start_model_evaluation(
        self, transformation_artifact: DataTransformationArtifact
    ) -> ModelEvaluationArtifact:
        logging.info("📊 Starting model evaluation")
        evaluator = ModelEvaluation(self.model_evaluation_config, transformation_artifact)
        return evaluator.initiate_model_evaluation()

    def run_pipeline(self):
        try:
            logging.info("🏁 Pipeline execution started")

            ingestion_artifact = self.start_data_ingestion()
            validation_artifact = self.start_data_validation(ingestion_artifact)

            if not validation_artifact.validation_status:
                raise Exception("❌ Data validation failed. Stopping pipeline.")

            transformation_artifact = self.start_data_transformation(ingestion_artifact)
            trainer_artifact = self.start_model_trainer(transformation_artifact)
            evaluation_artifact = self.start_model_evaluation(transformation_artifact)

            logging.info(f"📈 Evaluation Score: {evaluation_artifact.silhouette_score}")
            logging.info("✅ Pipeline execution completed successfully")

        except Exception as e:
            raise USvisaException(e, sys)





