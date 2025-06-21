# main.py or app.py

from src.pipline.training_pipeline import TrainPipeline  # ✅ Corrected import spelling

if __name__ == "__main__":
    obj = TrainPipeline()
    obj.run_pipeline()
