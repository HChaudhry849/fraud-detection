from tabpfn import TabPFNClassifier
from sklearn.pipeline import Pipeline
import vectorize_data
import pandas as pd

class TrainModel:
    def __init__(self, vd_instance):
        # FIX: Use the argument passed (vd_instance) instead of a global variable
        self.vd = vd_instance 
        
        # We fetch the ALREADY vectorized data for training the 'brain'
        self.X_train = vd_instance.X_train_vectorized
        self.y_train = vd_instance.y_train
        self.x_test = vd_instance.X_test_vectorized
        self.model = None

    def load_model(self):
        # Initializing the TabPFN brain
        self.model = TabPFNClassifier(device='cpu')
        return self.model

    def feed_model(self):
        # TabPFN works best when it knows feature names. 
        # If vectorize_data returned a numpy array, we convert it back to a DataFrame 
        # using the names from our 'Handbook' (transformer).
        if not isinstance(self.X_train, pd.DataFrame):
            cols = self.vd.transformer.get_feature_names_out()
            self.X_train = pd.DataFrame(self.X_train, columns=cols)
            
        self.model.fit(self.X_train, self.y_train)
        return self.model
    
    def predict(self):
        # We ensure test data also has names if it's currently a numpy array
        if not isinstance(self.x_test, pd.DataFrame):
            cols = self.vd.transformer.get_feature_names_out()
            self.x_test = pd.DataFrame(self.x_test, columns=cols)
            
        self.predictedResult = self.model.predict(self.x_test)
        return self.predictedResult
    
    def seeResult(self):
        print("Predictions:", self.predictedResult)
    
    def create_unified_model(self):
        # FIX: We glue the DICTIONARY (transformer) to the BRAIN (model).
        # This Pipeline is what we will save to MLflow.
        # When Flask calls pipeline.predict(raw_data), it will:
        # 1. Use the transformer to turn 5 columns into 7.
        # 2. Pass those 7 columns to the model.
        model_pipeline = Pipeline([
            ("vectorizer", self.vd.transformer), 
            ("classifier", self.model)
        ])
        return model_pipeline

# --- Execution Block ---
if __name__ == "__main__":
    vd_instance = vectorize_data.VectorizeData().prepare() # Prepares data & fits transformer
    tm = TrainModel(vd_instance)                          # Links the data
    tm.load_model()                                       # Creates the TabPFN brain
    tm.feed_model()                                       # Trains brain on vectorized data
    tm.predict()                                          # Tests brain
    tm.seeResult()