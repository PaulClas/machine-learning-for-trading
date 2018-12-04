from pathlib import Path
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date, datetime
from time import time

path = Path('reports')

# df = pd.read_csv('report_index.csv').rename(columns=str.lower)
# print(df.info())

start = time()
for report in path.glob('*.txt'):
    report.read_text()
print(time() - start)
