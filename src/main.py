import data_pipeline as dp
import vectorize_data as vd
import train_model as tm
import evaluation as ev 
import mlflow
# 1. Point MLflow to your database file
mlflow.set_tracking_uri("sqlite:///mlflow.db")
#2. Name the project 
mlflow.set_experiment("Fraud_Detection_System")

class Main:

    def __init__(self):
        with mlflow.start_run():
            self.DP = dp.DataPipeline()
        # 1. Setup Data
            self.VP = vd.VectorizeData().prepare() 
            # 2. Setup/Train Model
            self.TM = tm.TrainModel(self.VP)
            self.TM.load_model()
            self.TM.feed_model()
            self.TM.predict()
        # 3. Evaluate - Pass the instances created above
            self.EV = ev.Evaluation(self.TM, self.VP)
            recall, f1 = self.EV.evaluate_model()
        # 4. LOG THE RESULTS TO MLFLOW
            mlflow.log_param("model_type", "TabPFN")
            mlflow.log_param("test_size", 0.2)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            print("Successfully logged run to MLflow!")

if __name__ == "__main__":
   mainapp = Main()