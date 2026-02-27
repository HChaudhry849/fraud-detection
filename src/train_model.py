import mlflow

# 1. Tell MLflow where to send the data
# Since your UI is running on http://127.0.0.1:5000, we point the code there
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 2. Create or select an Experiment
# This groups your work so it doesn't get lost
mlflow.set_experiment("Fraud_Detection_Experiment")

# 3. Start a "Run"
# Think of this as pressing 'Record' on a stopwatch
with mlflow.start_run(run_name="Connection_Test"):
    
    # 4. Log a "Parameter" (The setup)
    # Usually, this would be a setting like 'learning_rate'
    mlflow.log_param("model_type", "Test_Skeleton")
    
    # 5. Log a "Metric" (The result)
    # We'll use a fake accuracy score of 95% just to see it appear
    mlflow.log_metric("accuracy", 0.95)
    
    print("Run complete! Check your MLflow dashboard.")