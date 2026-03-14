from flask import Flask, request, jsonify
import pandas as pd
import mlflow
import os
import platform
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

load_dotenv()
##################################################################
#This will not be needed for cloud, only needed for docker due to windows -> linux
#in cloud we simply work using the db url hosted on server 
###################################################################
# --- DYNAMIC MLFLOW LOGIC WITH DEEP DEBUG ---
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
model_alias = os.environ.get("MODEL_VERSION", "production") 

mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

print("--- STARTING SYSTEM AUDIT ---")
base_path = "/app/mlruns"

if os.path.exists(base_path):
    print(f"Audit of {base_path}:")
    for root, dirs, files in os.walk(base_path):
        print(f"  Folder: {root} | Contains Files: {files}")
else:
    print(f"CRITICAL ERROR: {base_path} does not exist!")

try:
    # 1. Get Registry Data
    model_data = client.get_model_version_by_alias("Fraud_Detection_Service", model_alias)
    run_id = model_data.run_id
    print(f"Targeting Run ID: {run_id}")

    # 2. Determine URI
    if platform.system() != "Windows":
        found_path = None
        for root, dirs, files in os.walk(base_path):
            if "MLmodel" in files:
                found_path = root
                break
        
        if found_path:
            model_uri = found_path
            print(f"--- SUCCESS: Found model at {model_uri} ---")
        else:
            model_uri = f"/app/mlruns/1/{run_id}/artifacts/model"
            print(f"--- ERROR: No MLmodel found, falling back to: {model_uri} ---")
    else:
        # Windows local development URI
        model_uri = f"models:/Fraud_Detection_Service@{model_alias}"
        print(f"--- Windows: Using Registry URI {model_uri} ---")

    # 3. CRITICAL: Actually Load the Model
    print(f"Attempting load from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    print("--- MODEL LOADED SUCCESSFULLY ---")

except Exception as e:
    print(f"FAILED TO LOAD MODEL: {e}")
    model = None

print("--- ENDING SYSTEM AUDIT ---")
##############################################################

app = Flask(__name__)

@app.route('/predict', methods=["POST"])
def predict():
    data = request.get_json()
    
    # Ensure data is a list (even if one row is sent)
    if isinstance(data, dict):
        data = [data]
        
    df = pd.DataFrame(data)
    
    # 1. CRITICAL: Reorder columns to match the training set EXACTLY
    # The ColumnTransformer is very picky about order.
    column_order = ["amount", "location", "type", "hour", "day_of_week"]
    df = df[column_order]
    
    try:
        # 2. Predict using the pipeline
        # The pipeline will automatically run vectorize_data on this 5-column df
        # and turn it into the 7 columns the classifier needs.
        prediction = model.predict(df)  
        myresults = []
        for x in prediction:
            if x == 0:
                result = "Not Fraud"
                myresults.append(result)
            elif x==1:
                result = "Fraud"
                myresults.append(result)

        return jsonify({'prediction': myresults})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/health")
def health():
    return "OK", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)