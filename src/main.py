import data_pipeline as dp
import vectorize_data as vd
import train_model as tm
import evaluation as ev 
import mlflow
from mlflow.models import infer_signature # Added for the fix

# 1. Point MLflow to your database file
mlflow.set_tracking_uri("sqlite:///mlflow.db")
# 2. Name the project 
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
            # 3. Evaluate
            self.EV = ev.Evaluation(self.TM, self.VP)
            passed, recall, f1 = self.EV.evaluate_model()
            
            # 4. LOG THE RESULTS
            mlflow.log_param("model_type", "TabPFN")
            mlflow.log_param("test_size", 0.2)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            print("Successfully logged run to MLflow!")

            # 5. THE MLOPS STEP: Glue and register
            if passed:
                # Get the unified Pipeline (Vectorizer + Brain)
                pipeline_model = self.TM.create_unified_model()
                
               # --- THE "HAPPY MEAL BOX" FIX ---

                # 1. We look at the "Questions" (The 5 things: Amount, Location, Type, Hour, Day)
                # 2. We look at the "Answer" (Is it Fraud? Yes or No)
                # 3. We write a "Label" (Signature) that says: 
                #    "To play with this toy, you must give me those 5 specific questions!"

                signature = infer_signature(self.VP.X_train, self.TM.predict())

                # Now, when you save the model, you aren't just saving a 'brain.'
                # You are saving the 'Brain' + 'The Translator' + 'The Instruction Manual' (Signature).
                
                # We also provide a small example of the raw input
                input_example = self.VP.X_train.iloc[[0]]

                mlflow.sklearn.log_model(
                    sk_model=pipeline_model,
                    artifact_path="fraud_model",
                    registered_model_name="Fraud_Detection_Service",
                    signature=signature,      # Tells MLflow the 5-column schema
                    input_example=input_example # Helps debug data types
                )
                # --- FIX ENDS HERE ---
                
                print("Model passed! Registered as a new version with Schema.")
            else:
                print("Model failed standards. Not registered.")

if __name__ == "__main__":
    mainapp = Main()