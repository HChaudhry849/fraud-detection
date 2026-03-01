import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

class VectorizeData:

    def __init__(self):
        self.X_train = None
        self.X_test = None  
        self.y_train = None
        self.y_test = None
        self.X = None
        self.Y = None
        self.PATH = 'data/processed/processed_data.parquet'

    def load_data(self):
        try:
            self.data = pd.read_parquet(self.PATH)
            print(self.data.dtypes)
        except Exception as e:
            raise RuntimeError("Failed to load file: {e}")
        
        return self.data
    
    #function handles timestamp column, breaks into hour and day_of_week, drops original column
    def preprocess_time(self):
        # Convert the string object to a formal datetime format
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        # Extract numeric features that the model can actually process
        self.data['hour'] = self.data['timestamp'].dt.hour
        self.data['day_of_week'] = self.data['timestamp'].dt.dayofweek
        
        # Now that we have the numbers, we drop the original object/datetime column
        # so the model doesn't try to perform math on a timestamp object
        self.data = self.data.drop(columns=['timestamp'])
        
        print("Timestamp converted to numeric 'hour' and 'day_of_week'.")

    #Split the data into x and y, Y is the target e.g. what the model needs to predict
    #Split the data into X, X is the data the model trains and to predict Y
    def split_data(self):
        #split all the columns required 
        self.X = self.data[["amount", "location", "type", "hour", "day_of_week"]]
        #important note: this column is already 0 or 1 therefore it does not need any vectorization 
        self.Y = self.data["is_fraud"]    
        #the test size is defined as 20% of the data and random shuffle means mixing up the rows of your data like a deck of cards before you deal them.
        #42 is a seed that ensures you get the exact same "random" shuffle every time you run the code.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.20, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def vectorize_data(self):
        #Column transformer allows us to apply different vectorization techniques quickly 
        #As we have different types of columns we need different vectorization techniques 
        #Vectorization is the process of turning data into a specific vocabulary that a computer can understand
        #After vectorization we end up with a dictionary e.g. London becomes 1 
        transformer = ColumnTransformer(transformers=[
            ('t1',StandardScaler(),['amount']),
            ('t2',OrdinalEncoder(),['location']),
            ('t3',OneHotEncoder(),['type']), 
            ('t4',StandardScaler(),['hour']), 
            ('t5',StandardScaler(),['day_of_week'])
        ],remainder='passthrough')
        
        #fit transform applies the rules e.g. for this column apply this vectorization, it uses the data and builds the rule
        self.X_train_vectorized = transformer.fit_transform(self.X_train)
        #here no rules are built, we simply take the test data that we will use later and use the existing dictionary, 
        #to change the test data into same vocabulary that the computer can understand
        self.X_test_vectorized = transformer.transform(self.X_test)

        #we return the saved vectorized data to use for training 
        return self.X_train_vectorized,  self.X_test_vectorized


vd = VectorizeData()
vd.load_data()
vd.preprocess_time()
vd.split_data()
vd.vectorize_data()


#Steps to Follow (Next Phase)
# 1. Load Data (Done)
# 2. Split Data (Done)
# 3. Vectorize Data (Done)
# 4. Load Model (Done)
# 5. Train Model (Done)
# 6. Evaluate Model (Done)
# 7. Experiment Tracking

#Steps to Follow (Phase 2)
# 1. Model Registration 
# 2. Versioning and Staging 
# 3. Artifact Logging 
# 4. Deployment 
