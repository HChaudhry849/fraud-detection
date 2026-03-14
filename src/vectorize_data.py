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
        # To store results for training
        self.X_train_vectorized = None
        self.X_test_vectorized = None

    def vectorize_data(self):
        """
        Learns the rules from X_train and prepares the translator (transformer).
        This is the method that turns 5 raw columns into the 7 numeric columns.
        """
        # 1. THE TRANSLATOR'S TOOLS:
        # We assign it to self.transformer so the rest of the project can access it.
        # Added handle_unknown='ignore' so the Flask app doesn't crash on new categories.
        self.transformer = ColumnTransformer(transformers=[
            ('t1', StandardScaler(), ['amount']),
            ('t2', OrdinalEncoder(), ['location']),
            ('t3', OneHotEncoder(handle_unknown='ignore'), ['type']), 
            ('t4', StandardScaler(), ['hour']), 
            ('t5', StandardScaler(), ['day_of_week'])
        ], remainder='passthrough')
        
        # 2. LEARNING THE NEIGHBORHOOD (The 'Fit' step):
        # The translator looks at the data and learns the rules.
        # It writes down: "In this shop, London is Code #1 and a 'Large' is 16oz."
        # This fills the 'Handbook' (self.transformer) with real information.
        self.X_train_vectorized = self.transformer.fit_transform(self.X_train)
        
        # 3. DOING THE WORK (The 'Transform' step):
        # Now that the Handbook is full, the translator uses those 
        # EXACT same rules to turn the test data into numbers.
        self.X_test_vectorized = self.transformer.transform(self.X_test)

        # We return the numbers for training, but the 'self.transformer' handbook 
        # is now stored in the class instance so we can glue it to the brain later.
        return self.X_train_vectorized, self.X_test_vectorized
    
    def prepare(self):
        """One method to rule them all - ensures data is ready."""
        self.load_data()
        self.preprocess_time()
        self.split_data()
        self.vectorize_data() # This now correctly populates self.transformer
        return self # Allows chaining

    def load_data(self):
        try:
            self.data = pd.read_parquet(self.PATH)
            print(f"Data loaded successfully. Dtypes:\n{self.data.dtypes}")
        except Exception as e:
            raise RuntimeError(f"Failed to load file: {e}")
        
        return self.data
    
    def preprocess_time(self):
        """Handles timestamp column, breaks into hour and day_of_week, drops original column."""
        # Convert the string object to a formal datetime format
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        
        # Extract numeric features that the model can actually process
        self.data['hour'] = self.data['timestamp'].dt.hour
        self.data['day_of_week'] = self.data['timestamp'].dt.dayofweek
        
        # Now that we have the numbers, we drop the original object/datetime column
        # so the model doesn't try to perform math on a timestamp object
        self.data = self.data.drop(columns=['timestamp'])
        
        print("Timestamp converted to numeric 'hour' and 'day_of_week'.")

    def split_data(self):
        """Split the data into X (features) and Y (target/is_fraud)."""
        # split all the columns required 
        self.X = self.data[["amount", "location", "type", "hour", "day_of_week"]]
        # important note: this column is already 0 or 1 therefore it does not need any vectorization 
        self.Y = self.data["is_fraud"]    
        
        # the test size is defined as 20% of the data and random shuffle means mixing up the rows.
        # 42 is a seed that ensures you get the exact same "random" shuffle every time.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.Y, test_size=0.20, random_state=42
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

#Steps to Follow (Next Phase)
# 1. Load Data (Done)
# 2. Split Data (Done)
# 3. Vectorize Data (Done)
# 4. Load Model (Done)
# 5. Train Model (Done)
# 6. Evaluate Model (Done)
# 7. Experiment Tracking (Done)
# 8. Model and Management Versioning & Registry (Done)
# 9. Flask App (Done)

#Steps to Follow (Phase 2)
# 1. Containerization (Data Pipeline, Training Pipeline, Inference Pipeline with Flask)
# 2. AWS (Learning)
# 3. Workflow Orchestration & Deployment (Putting it all together )
# 4. CI/CD (See it run automatically and Monitor)
