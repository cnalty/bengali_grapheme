import pandas as pd

table = pd.read_parquet("train_image_data_0.parquet")
# image size (137, 236)
print(table.head())
