from flask import Flask, request, jsonify
import pandas as pd
import mlflow
import os

# Get project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# Point to the correct DB
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(project_root, 'src', 'mlflow.db')}")
# ===== Load your model =====
model = mlflow.sklearn.load_model("models:/Fraud_Detection_Service@production")

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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)