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
        # Think of this as an empty 'Translator Handbook.'
        # Right now, it's blank. We are just making a spot on the shelf 
        # to keep it so the shop (our code) doesn't lose it.
        self.transformer = None 
    
    def vectorize_data(self):
        # 1. THE TRANSLATOR'S TOOLS:
        # Here we decide HOW we will translate. 
        # 'StandardScaler' is for measuring amounts (scoops of sugar).
        # 'OneHotEncoder' is for labels (Strawberry vs. Banana).
        self.transformer = ColumnTransformer(transformers=[
            ('t1', StandardScaler(), ['amount']),
            ('t2', OrdinalEncoder(), ['location']),
            ('t3', OneHotEncoder(), ['type']), 
            ('t4', StandardScaler(), ['hour']), 
            ('t5', StandardScaler(), ['day_of_week'])
        ], remainder='passthrough')
        
        # 2. LEARNING THE NEIGHBORHOOD (The 'Fit' step):
        # The translator looks at the data and learns the rules.
        # It writes down: "In this shop, London is Code #1 and a 'Large' is 16oz."
        # This fills the 'Handbook' with real information.
        self.X_train_vectorized = self.transformer.fit_transform(self.X_train)
        # 3. DOING THE WORK (The 'Transform' step):
        # Now that the Handbook is full, the translator uses those 
        # EXACT same rules to turn the test data into numbers.
        self.X_test_vectorized = self.transformer.transform(self.X_test)

        # We return the numbers, but the 'self.transformer' handbook 
        # stays safe inside this class so we can glue it to the brain la
        return self.X_train_vectorized, self.X_test_vectorized
    
    def prepare(self):
        """One method to rule them all - ensures data is ready."""
        self.load_data()
        self.preprocess_time()
        self.split_data()
        self.vectorize_data()
        return self # Allows chaining

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


#Steps to Follow (Next Phase)
# 1. Load Data (Done)
# 2. Split Data (Done)
# 3. Vectorize Data (Done)
# 4. Load Model (Done)
# 5. Train Model (Done)
# 6. Evaluate Model (Done)
# 7. Experiment Tracking (Done)
# 8. Model and Management Versioning & Registry (Done)

#Steps to Follow (Phase 2)
# 1. Containerization (Data Pipeline, Training Pipeline, Inference Pipeline with Flask)
# 2. AWS (Learning)
# 3. Workflow Orchestration & Deployment (Putting it all together )
# 4. CI/CD (See it run automatically and Monitor)
