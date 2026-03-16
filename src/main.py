# mlflow ui --backend-store-uri sqlite:///mlflow.db 
import data_pipeline as dp
import vectorize_data as vd
import train_model as tm
import evaluation as ev 
import mlflow
from mlflow.models import infer_signature 
import os

# --- ENVIRONMENT-AWARE MLFLOW CONFIG ---
# 1. Check if we are in Docker (Variable provided by docker-compose)
# 2. If not, calculate the local Windows path
env_uri = os.getenv("MLFLOW_TRACKING_URI")

if env_uri:
    mlflow.set_tracking_uri(env_uri)
    # Explicitly tell Docker where the artifacts go so it matches the volume
    os.environ["MLFLOW_ARTIFACT_ROOT"] = "/app/mlruns" 
    print(f"--- Running in DOCKER mode: Using {env_uri} ---")
else:
    # We are on Windows Local
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(script_dir, "..", "database"))
    local_uri = f"sqlite:///{os.path.join(base_path, 'mlflow.db')}"
    mlflow.set_tracking_uri(local_uri)
    
    # Ensure local directory exists for Windows
    os.makedirs(base_path, exist_ok=True)
    
    # Sync Artifact path for local runs
    mlruns_path = os.path.join(base_path, "mlruns")
    os.environ["MLFLOW_ARTIFACT_ROOT"] = mlruns_path 
    print(f"--- Running in LOCAL mode: Using {local_uri} ---")

# 3. Name the project 
mlflow.set_experiment("Fraud_Detection_System")

class Main:
    def __init__(self):
        # start_run will now use the URI set above automatically
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
                
                # Create the Signature (The Schema Label)
                signature = infer_signature(self.VP.X_train, self.TM.predict())

                # Example for debugging
                input_example = self.VP.X_train.iloc[[0]]

                # Log and Register the model
                mlflow.sklearn.log_model(
                    sk_model=pipeline_model,
                    artifact_path="fraud_model",
                    registered_model_name="Fraud_Detection_Service",
                    signature=signature,
                    input_example=input_example
                )
                
                print("Model passed! Registered as a new version with Schema.")
            else:
                print("Model failed standards. Not registered.")

if __name__ == "__main__":
    mainapp = Main()