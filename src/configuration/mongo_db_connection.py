import pymongo
import certifi
import sys
from src.constants import DATABASE_NAME, MONGODB_URL
from src.exception import USvisaException
from src.logger import logging

ca = certifi.where()

class MongoDBClient:
    client = None

    def __init__(self, database_name=DATABASE_NAME):
        try:
            if MongoDBClient.client is None:
                if MONGODB_URL is None:
                    raise ValueError("MongoDB URL is not set.")
                MongoDBClient.client = pymongo.MongoClient(MONGODB_URL, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            logging.info("✅ MongoDB connection successful.")
        except Exception as e:
            raise USvisaException(e, sys)

