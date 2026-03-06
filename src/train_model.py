from tabpfn import TabPFNClassifier
from sklearn.pipeline import Pipeline
import vectorize_data

vd = vectorize_data.VectorizeData()

class TrainModel:

    def __init__(self, vd):
        # We "give" the class the data it needs right when it's born
        self.X_train = vd.X_train_vectorized
        self.y_train = vd.y_train
        self.x_test = vd.X_test_vectorized
        self.model = None
        self.vd = vd_instance # Store the whole instance so we can get the transformer later

    def load_model(self):
        self.model = TabPFNClassifier(device='cpu')
        return self.model

    def feed_model(self):
        self.model.fit(self.X_train,  self.y_train)
        return self.model
    
    def predict(self):
        self.predictedResult = self.model.predict(self.x_test)
        return self.predictedResult
    
    def seeResult(self):
        print(self.predictedResult)
    
    def create_unified_model(self):
        # We glue the DICTIONARY (transformer) to the BRAIN (model)
        model_pipeline = Pipeline([
            ("vectorizer", self.vd.transformer), 
            ("classifier", self.model)
        ])
        return model_pipeline


vd_instance = vd.prepare()      # Prepares the data
tm = TrainModel(vd_instance)    # Links the data
tm.load_model()                 # NEW: This actually creates the TabPFNClassifier
tm.feed_model()                 # Now this works because self.model is no longer None
tm.predict()
tm.seeResult()