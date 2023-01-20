import pandas as pd
import os

if __name__ == "__main__":

    working_dir = os.path.dirname(__file__)
    data_path = "data/tabular_data/listing.csv"
    path = os.path.join(working_dir, data_path)
    listing_df = pd.read_csv(path)