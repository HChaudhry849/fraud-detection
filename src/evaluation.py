from sklearn.metrics import f1_score, recall_score
import train_model
import vectorize_data

tm = train_model.TrainModel()
vd = vectorize_data.VectorizeData()

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
        else:
            print("Model failed evaluation")
        
        return self.recall, self.f_score

ev = Evaluation()
ev.evaluate_model()