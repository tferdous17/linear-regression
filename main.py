import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('MOCK_DATA.csv')

def mean_squared_error(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].x
        y = points.iloc[i].y
        total_error += (y - (m * x + b)) ** 2
    total_error / float(len(points))
