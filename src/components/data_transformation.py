import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.logger import logging
from src.exception import USvisaException
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from src.utils.main_utils import save_object, read_yaml_file
from src.constants import SCHEMA_FILE_PATH

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        try:
            num_features = self.schema_config["num_features"]
            oh_columns = self.schema_config["oh_columns"]

            transformer = ColumnTransformer([
                ("num", StandardScaler(), num_features),
                ("cat", OneHotEncoder(handle_unknown='ignore'), oh_columns)
            ])

            return transformer
        except Exception as e:
            raise USvisaException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            # Load data
            df = pd.read_csv(self.data_ingestion_artifact.feature_store_file_path)

            # Drop unwanted columns
            df.drop(columns=self.schema_config["drop_columns"], inplace=True)

            # Separate input features
            input_features = self.schema_config["transform_columns"] + self.schema_config["oh_columns"]

            input_df = df[input_features]

            # Fit-transform
            transformer = self.get_data_transformer_object()
            transformed_array = transformer.fit_transform(input_df)

            # Save transformer
            os.makedirs(os.path.dirname(self.data_transformation_config.transformer_object_path), exist_ok=True)
            save_object(self.data_transformation_config.transformer_object_path, transformer)

            # Save transformed data
            transformed_df = pd.DataFrame(transformed_array.toarray() if hasattr(transformed_array, "toarray") else transformed_array)
            transformed_df.to_csv(self.data_transformation_config.transformed_data_path, index=False)

            logging.info("✅ Data Transformation completed.")

            return DataTransformationArtifact(
                transformed_data_path=self.data_transformation_config.transformed_data_path,
                transformer_object_path=self.data_transformation_config.transformer_object_path
            )

        except Exception as e:
            raise USvisaException(e, sys)
