import pandas as pd

filename = 'pre_XIIoTID.csv'

data = pd.read_csv(filename, low_memory = False)