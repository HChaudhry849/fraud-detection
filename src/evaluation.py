from sklearn.metrics import f1_score, recall_score

class Evaluation():

    def __init__(self, tm, vd):
        self.model = tm.model
        self.test = vd.y_test
        self.result = tm.predictedResult

    def evaluate_model(self):
        THRESHOLDS = {
            "recall": 0.80, 
            "f1": 0.80
        }

        self.recall = recall_score(self.test,  self.result)
        print(self.recall)

        self.f_score = f1_score(self.test,  self.result)
        print(self.f_score)

        if self.recall >= THRESHOLDS["recall"] and self.f_score >= THRESHOLDS["f1"]:
            print("Model passed evaluation")
            passed = True
        else:
            print("Model failed evaluation")
            passed = False
        
        return passed, self.recall, self.f_score

# WRAP THIS IN THE PROTECTIVE BLOCK
if __name__ == "__main__":
    import vectorize_data
    import train_model
    
    # 1. Prepare Data
    vd_instance = vectorize_data.VectorizeData().prepare()
    
    # 2. Train Model
    tm_instance = train_model.TrainModel(vd_instance)
    tm_instance.load_model()
    tm_instance.feed_model()
    tm_instance.predict()
    
    # 3. Evaluate
    ev = Evaluation(tm_instance, vd_instance)
    ev.evaluate_model()