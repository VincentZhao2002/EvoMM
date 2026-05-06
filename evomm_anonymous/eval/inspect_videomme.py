import pandas as pd
import os

try:
    df = pd.read_parquet("./datasets/video-mme/videomme/test-00000-of-00001.parquet")
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFirst row:")
    print(df.iloc[0].to_dict())
    
    # Check if we can find where the videos are
    # Maybe video_id corresponds to the filename?
    print("\nSample video_ids:", df['video_id'].head().tolist())
    print("\nSample videoIDs:", df['videoID'].head().tolist())
    
except Exception as e:
    print(f"Error: {e}")
