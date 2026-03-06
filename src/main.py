#mlflow ui --backend-store-uri sqlite:///mlflow.db

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
        # 1. Setup Data, runs entire class 
            self.VP = vd.VectorizeData().prepare() 
            # 2. Setup/Train Model
            self.TM = tm.TrainModel(self.VP)
            self.TM.load_model()
            self.TM.feed_model()
            self.TM.predict()
        # 3. Evaluate - Pass the instances created above
            self.EV = ev.Evaluation(self.TM, self.VP)
            passed, recall, f1 = self.EV.evaluate_model()
        # 4. LOG THE RESULTS TO MLFLOW
            mlflow.log_param("model_type", "TabPFN")
            mlflow.log_param("test_size", 0.2)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            print("Successfully logged run to MLflow!")

        # 5. THE MLOPS STEP: Only glue and register if it's a winner!
            if passed:
             # We ask the Trainer to glue the Dictionary to the Brain
                low_code_robot = self.TM.create_unified_model()
                
                # We save that one "Unified Box" to the Registry
                mlflow.sklearn.log_model(
                    sk_model=low_code_robot,
                    artifact_path="fraud_model",
                    registered_model_name="Fraud_Detection_Service"
                )
                print("Model passed! Registered as a new version.")
            else:
                print("Model failed standards. Not registered.")

if __name__ == "__main__":
   mainapp = Main()