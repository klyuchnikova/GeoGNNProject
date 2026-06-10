import pandas as pd

class TemporalSplitter:
    def __init__(self, train_ratio=0.8, val_ratio=0.1):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def split(self, df):
        train_parts = []
        val_parts = []
        test_parts = []
        
        for _, user_df in df.sort_values("timestamp").groupby("user_id"):
            n = len(user_df)
            train_end = int(n * self.train_ratio)
            val_end = int(n * (self.train_ratio + self.val_ratio))
            
            train_parts.append(user_df.iloc[:train_end])
            val_parts.append(user_df.iloc[train_end:val_end])
            test_parts.append(user_df.iloc[val_end:])
        
        return pd.concat(train_parts), pd.concat(val_parts), pd.concat(test_parts)