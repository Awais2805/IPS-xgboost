
import pandas as pd
import numpy as np
import sklearn
network_traffic_dataset = pd.read_csv("dataset/cic.csv")

print(network_traffic_dataset.head(5))
print(network_traffic_dataset.info())
print(network_traffic_dataset.describe().T)


print(f'Size of CIC-IDS2018 before dropping duplicates: {network_traffic_dataset.shape}')
print(f'Number of duplicate rows: {network_traffic_dataset.duplicated().sum()}')
print(network_traffic_dataset[network_traffic_dataset.duplicated()])
network_traffic_dataset = network_traffic_dataset.drop_duplicates()
print(f'Size of CIC-IDS2018 after dropping duplicates: {network_traffic_dataset.shape}')
print(network_traffic_dataset.columns.tolist())
# print(network_traffic_dataset['Label'].value_counts())
    