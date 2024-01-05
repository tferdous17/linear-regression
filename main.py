import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('MOCK_DATA.csv')

# mean squared error = summation[0..len(points)] (predicted val - actual val)^2
def mean_squared_error(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].x
        y = points.iloc[i].y
        total_error += (y - (m * x + b)) ** 2
    total_error / float(len(points))

# optimizes/minimizes the params m, b to achieve a low overall error (cost) for the function
def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].x
        y = points.iloc[i].y

        # partial derivatives
        m_gradient += -(2 / n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / n) * (y - (m_now * x + b_now))

    # L = learning rate
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b


m = 0
b = 0
L = 0.001
epochs = 500 # num of iterations

for i in range(epochs):
    if i % 50 == 0:
        print(f'epoch: {i}')
    m, b = gradient_descent(m, b, data, L)

print(m, b)
plt.scatter(data.x, data.y, color="black")
plt.plot(list(range(0, 100)), [m * x + b for x in range(0, 100)], color="red")
plt.show()

# note: plot not properly working, something with scalar add overflow