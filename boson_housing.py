"""Load the Boston dataset and examine its target (label) distribution."""

import numpy as np
import pylab as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

import pandas as pd
# Citation #1
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import NearestNeighbors


def load_data():
    """Load the Boston dataset."""

    boston = datasets.load_boston()
    return boston


def explore_city_data(city_data):
    """Calculate the Boston housing statistics."""

    housing_prices      = city_data.target
    housing_features    = city_data.data
    
    # Citation #2
    data_size           = housing_prices.shape[0]
    no_of_features      = housing_features.shape[1]
    min_price           = np.min(housing_prices)
    max_price           = np.max(housing_prices)
    mean_price          = np.mean(housing_prices)
    median_price        = np.median(housing_prices)
    standard_dev        = np.std(housing_prices)
    
    # Citation #3
    city_stats          = [data_size, no_of_features, min_price, max_price, mean_price, 
                           median_price, standard_dev]
    city_labels         = ['Data Size', 'Number of Features', 'Minimum Price', 'Maximum Price',
                           'Mean Price', 'Median Price', 'Standard Deviation']

    city_series         = pd.Series(city_stats, index=city_labels)

    return city_series


def split_data(city_data):
    """Randomly shuffle the sample set. Divide it into 70 percent training and 30 percent testing data."""

    X, y = city_data.data, city_data.target

    # Citation #4
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

    pd_data             = [X_train, y_train, X_test, y_test]
    pd_labels           = ['X Training', 'y Training', 'X Testing', 'y Testing']
    split_data_series   = pd.Series(pd_data, index=pd_labels)

    return split_data_series    


def performance_metric(label, prediction):
    """Calculate and return the appropriate error performance metric."""

    # Citation #5
    mse = mean_squared_error(label, prediction)

    return mse


def learning_curve(depth, X_train, y_train, X_test, y_test):
    """Calculate the performance of the model after a set of training data."""

    sizes = np.round(np.linspace(1, len(X_train), 50))
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    print "Decision Tree with Max Depth: "
    print depth

    for i, s in enumerate(sizes):

        regressor = DecisionTreeRegressor(max_depth=depth)
        regressor.fit(X_train[:s], y_train[:s])

        train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    learning_curve_graph(sizes, train_err, test_err)


def learning_curve_graph(sizes, train_err, test_err):
    """Plot training and test error as a function of the training size."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Training Size')
    pl.plot(sizes, test_err, lw=2, label = 'test error')
    pl.plot(sizes, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Training Size')
    pl.ylabel('Error')
    pl.show()


def model_complexity(X_train, y_train, X_test, y_test):
    """Calculate the performance of the model as model complexity increases."""

    print "Model Complexity: "

    max_depth = np.arange(1, 25)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        regressor = DecisionTreeRegressor(max_depth=d)

        regressor.fit(X_train, y_train)

        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    model_complexity_graph(max_depth, train_err, test_err)


def model_complexity_graph(max_depth, train_err, test_err):
    """Plot training and test error as a function of the depth of the decision tree learn."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Max Depth')
    pl.plot(max_depth, test_err, lw=2, label = 'test error')
    pl.plot(max_depth, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Max Depth')
    pl.ylabel('Error')
    pl.show()


def fit_predict_model(city_data, x):
    """Find and tune the optimal model. Make a prediction on housing data."""

    X, y = city_data.data, city_data.target

    regressor = DecisionTreeRegressor()
    # Citation #5
    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}
    # Citation #6
    acc_scorer = make_scorer(mean_squared_error, greater_is_better=False)

    # Citation #7
    reg = GridSearchCV(regressor, parameters, scoring=acc_scorer)
    reg.fit(X, y)

    # Citation #8
    print "Best estimator: "
    print reg.best_estimator_
    print "Final Model: "
    print reg.fit(X, y)

    x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
    y = reg.predict(x)
    
    print "House: " + str(x)
    print "Prediction: " + str(y)
    
    
def find_three_neighbor_average(x):
    X = datasets.load_boston().data
    
    def find_nearest_neighbor_indexes(x, X):
        neighbor = NearestNeighbors(n_neighbors=10)
        neighbor.fit(X)
        distance, indexes = neighbor.kneighbors(x)
        return indexes

    indexes = find_nearest_neighbor_indexes(x, X)
    sum_prices = []
    
    for i in indexes:
        sum_prices.append(datasets.load_boston().target[i])
    
    neighbor_avg = np.mean(sum_prices)
    
    print "Nearest Neighbors average: " + str(neighbor_avg)


def main():
    """Analyze the Boston housing data. Evaluate and validate the
    performanance of a Decision Tree regressor on the housing data.
    Fine tune the model to make prediction on unseen data."""

    city_data = load_data()

    explore_city_data(city_data)

    X_train, y_train, X_test, y_test = split_data(city_data)

    max_depths = range(1, 11)
    for max_depth in max_depths:
        learning_curve(max_depth, X_train, y_train, X_test, y_test)

    model_complexity(X_train, y_train, X_test, y_test)

    # I added this; it was probably missing.
    fit_predict_model(city_data, x)
    
    find_three_neighbor_average(x)


if __name__ == "__main__":
    main()
