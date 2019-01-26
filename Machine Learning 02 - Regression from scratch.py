from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

def create_dataset(number_of_datapoints, slope, variance):
    val = 1
    xs = [i for i in range(number_of_datapoints)]
    ys = [slope*x + random.random()*variance for x in xs]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

style.use('fivethirtyeight')

# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64) # sample data


def best_fit_line(x, y):
    m = (mean(x)*mean(y)-mean(x*y))/(mean(x)**2 - mean(x**2)) # formulas for linear regression gradient and intercept
    b = mean(y) - m*mean(x)
    return m, b


# squared error is the distance from the point to best fit line squared
# penalizes outliers because squaring large errors magnifies error
# r^2 = 1 - SE(y hat)/SE(y mean)
# y hat: best fit line, SE: squared error
# the closer r is to 1, the better fit the line


def squared_err(y_original, y_line):
    return sum((y_original - y_line)**2) # sum since y_original and y_line is a list


def coefficient_of_determination(y_original, y_line): # r
    y_mean_line = [mean(y_original) for y in y_original]
    squared_error_regr = squared_err(y_original, y_line)
    squared_error_y_mean = squared_err(y_original, y_mean_line)

    return 1 - squared_error_regr/squared_error_y_mean


xs, ys = create_dataset(50, 4, 50)
m, b = best_fit_line(xs, ys)
print(m, b)

regression_line = [m*x + b for x in xs] # one liner equal to:
# for x in xs:
#     regression_line.append = m*x + b

predict_x = 8
predict_y = predict_x*m + b # trying to predict when x=8

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)


