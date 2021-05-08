import os
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def load_points_from_file(filename):
    """Loads 2d points from a csv called filename
    Args:
        filename : Path to .csv file
    Returns:
        (xs, ys) where xs and ys are a numpy array of the co-ordinates.
    """
    points = pd.read_csv(filename, header=None)
    return points[0].values, points[1].values


def view_data_segments(xs, ys):
    """Visualises the input file with each segment plotted in a different colour.
    Args:
        xs : List/array-like of x co-ordinates.
        ys : List/array-like of y co-ordinates.
    Returns:
        None
    """
    assert len(xs) == len(ys)
    assert len(xs) % 20 == 0
    len_data = len(xs)
    num_segments = len_data // 20
    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
    plt.set_cmap('Dark2')
    plt.scatter(xs, ys, c=colour)
    plt.show()


def leastSquaresLin(xs, ys):
    ones = np.ones(xs.shape)
    x2 = np.column_stack((ones, xs))
    return np.linalg.inv(x2.T.dot(x2)).dot(x2.T).dot(ys)


def leastSquaresPoly(xs, ys):
    ones = np.ones(xs.shape)
    x2 = np.column_stack((ones, xs, xs ** 2, xs ** 3))
    return np.linalg.inv(x2.T.dot(x2)).dot(x2.T).dot(ys)

def leastSquaresUnknown(xs, ys):
    ones = np.ones(xs.shape)
    x2 = np.column_stack((ones, np.sin(xs)))
    return np.linalg.inv(x2.T.dot(x2)).dot(x2.T).dot(ys)

# Returns squared error between calculated and real values
def squaredError(calculated, real):
    error = 0
    for i in range(len(calculated)):
        error += (calculated[i] - real[i])**2
    return error


# Returns the error for a linear function. a, b refer to ax + b
def linError(a, b, xs, ys):
    calculatedY = [a*x + b for x in xs]
    return squaredError(calculatedY, ys)


def polyError(a, b, c, d, xs, ys):
    calculatedY = [a*(x ** 3) + b*(x ** 2) + c*x + d for x in xs]
    return squaredError(calculatedY, ys)

def unknownError(a, b, xs, ys):
    calculatedY = [a*np.sin(x) + b for x in xs]
    return squaredError(calculatedY, ys)

def crossValidation(segments, xs, ys):
    #lines[j] has all the test results for the jth segment
    lines = [[]]*50
    for i in range(50):
        rng = np.random.default_rng()
        randomNums = rng.choice(20, size=5, replace=False)
        x_train = list(range(20))
        y_train = list(range(20))
        x_test = list(range(5))
        y_test = list(range(5))
        for j in range(5):
            x_test[j] = x_train[randomNums[j]]
            y_test[j] = y_train[randomNums[j]]
        for j in range(5):
            x_train.remove(randomNums[j])
            y_train.remove(randomNums[j])
        
        totalError = 0
        for j in range(int(round(segments))):
            #crossover is fine since the new function should start at the point that the old one ends
            xseg = np.array([xs[j*20 + index] for index in x_train])
            yseg = np.array([ys[j*20 + index] for index in y_train])
            xsegTest = np.array([xs[j*20 + index] for index in x_test])
            ysegTest = np.array([ys[j*20 + index] for index in y_test])
            b, a = leastSquaresLin(xseg, yseg)
            errorL = linError(a, b, xsegTest, ysegTest)
            f, e, d, c = leastSquaresPoly(xseg, yseg)
            errorP = polyError(c, d, e, f, xsegTest, ysegTest)
            ub, ua = leastSquaresUnknown(xseg, yseg)
            errorU = unknownError(ua, ub, xsegTest, ysegTest)
            error = min(errorL, errorP, errorU)
            totalError += error
            if error == errorL:
                lines[j].append(["Linear", a, b, error])
            elif error == errorP:
                lines[j].append(["Poly", c, d, e, f, error])
            else:
                lines[j].append(["Unknown", ua, ub, error])
    goodLines = smallestError(lines, segments)
    return goodLines



def smallestError(lines, segments):
    goodLines = [0]*int(round(segments))
    for j in range(int(round(segments))):
        error = sys.maxsize
        index = 0
        for i in range(50):
            if lines[i][j][0] == "Linear":
                if lines[i][j][3] < error:
                    index = i
                    error = lines[j][i][3]
            elif lines[i][j][0] == "Poly":
                if lines[i][j][5] < error:
                    index = i
                    error = lines[i][j][5]
            else:
                if lines[i][j][3] < error:
                    index = i
                    error = lines[i][j][3]
        goodLines[j] = lines[index][j]
    return goodLines




file = sys.argv[1]
xs, ys = load_points_from_file(file)
len(xs)
segments = len(xs)/20
goodLines = crossValidation(segments, xs, ys)

TError = 0
for i in range(int(round(segments))):
    xseg = xs[i*20:(i+1)*20]
    if goodLines[i][0] == "Linear":
        print("Lin")
        a = goodLines[i][1]
        b = goodLines[i][2]
        plt.plot(xseg, [a*x + b for x in xseg], 'r-', lw = 2)
        TError += goodLines[i][3]
    elif goodLines[i][0] == "Poly":
        print("Poly")
        c = goodLines[i][1]
        d = goodLines[i][2]
        e = goodLines[i][3]
        f = goodLines[i][4]
        plt.plot(xseg, [c*(x ** 3) + d*(x ** 2) + e*x + f for x in xseg], 'r-', lw = 2)
        TError += goodLines[i][5]
    else:
        print("Unknown")
        ua = goodLines[i][1]
        ub = goodLines[i][2]
        plt.plot(xseg, [ua*np.sin(x) + ub for x in xseg], 'r-', lw = 2)
        TError += goodLines[i][3]

print(TError)

if (len(sys.argv) > 2):
    if (sys.argv[2] == '--plot'):
        view_data_segments(xs, ys)