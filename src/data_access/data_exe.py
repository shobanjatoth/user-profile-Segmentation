from src.configuration.mongo_db_connection import MongoDBClient
from src.exception import USvisaException
import pandas as pd
import sys
from typing import Optional
import numpy as np

class USvisaData:
    def __init__(self):
        try:
            self.mongo_client = MongoDBClient()
        except Exception as e:
            raise USvisaException(e, sys)

    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        try:
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client.client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)
            df.replace({"na": np.nan}, inplace=True)
            return df
        except Exception as e:
            raise USvisaException(e, sys)
