import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_metadata(data_path):
    """Load metadata about audio files"""
    speakers = [f for f in os.listdir(data_path) 
                if os.path.isdir(os.path.join(data_path, f)) 
                and not f.startswith("_") 
                and not f.lower().startswith("other")]
    
    data_info = []
    for spk in speakers:
        folder = os.path.join(data_path, spk)
        files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        
        for f in files:
            file_path = os.path.join(folder, f)
            data_info.append([spk, file_path])
    
    return pd.DataFrame(data_info, columns=["speaker", "file_path"])

def split_dataset(df, test_size=0.2, val_size=0.2, random_state=42):
    """Split dataset into train, validation, and test sets"""
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['speaker']
    )
    
    # Second split: separate validation set from train
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=random_state, 
        stratify=train_val_df['speaker']
    )
    
    return train_df, val_df, test_df
