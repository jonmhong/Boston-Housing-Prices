"""Load the Boston dataset and examine its target (label) distribution."""

import numpy as np
import pylab as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
import time

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import NearestNeighbors

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import xgboost as xgb
import deap


def explore_city_data(city_data):
    """Calculate the Boston housing statistics."""

    # X and y
    housing_prices = city_data.target
    housing_features = city_data.data
    
    # statistical values
    data_size = housing_prices.shape[0]
    no_of_features = housing_features.shape[1]
    min_price = np.min(housing_prices)
    max_price = np.max(housing_prices)
    mean_price = np.mean(housing_prices)
    median_price = np.median(housing_prices)
    standard_dev = np.std(housing_prices)
    
    # creating Series
    city_stats = [data_size, no_of_features, min_price, max_price, mean_price, 
                           median_price, standard_dev]
    city_labels = ['Data Size', 'Number of Features', 'Minimum Price', 'Maximum Price',
                   'Mean Price', 'Median Price', 'Standard Deviation']

    city_series = pd.Series(city_stats, index=city_labels)

    return city_series


def split_data(city_data):
    """Randomly shuffle the sample set. Divide it into 70 percent training and 30 percent testing data."""

    X, y = city_data.data, city_data.target
    
    # random state is arbitrarily set to 12
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)


    pd_data = [X_train, y_train, X_test, y_test]
    pd_labels = ['X Training', 'y Training', 'X Testing', 'y Testing']
    split_data_series = pd.Series(pd_data, index=pd_labels)

    return split_data_series    


def performance_metric(label, prediction):
    """Calculate and return the performance, using the mean squared error metric."""
    mse = mean_squared_error(label, prediction)

    return mse


def learning_curve(depth, X_train, y_train, X_test, y_test):
    """Calculate the performance improvement of the model, as training size increases."""
    
    # create 50 equally spaced markers for the the graph's X axis
    sizes = np.round(np.linspace(1, len(X_train), 50))
    # create 50 open bins to fill in the training and test errors
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    print "Decision Tree with Max Depth: "
    print depth

    for i, s in enumerate(sizes):
        
        # train classifier and test on each level of depth complexity
        regressor = DecisionTreeRegressor(max_depth=depth)
        regressor.fit(X_train[:s], y_train[:s])
        
        # fill in the training and test error 
        train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # create the learning curve graph, using the calculated information
    learning_curve_graph(sizes, train_err, test_err)
    
    return test_err[-1]


def learning_curve_graph(sizes, train_err, test_err):
    """Plot training and test error as a function of the training size."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Training Size')
    
    pl.plot(sizes, test_err, lw=2, label='test error')
    pl.plot(sizes, train_err, lw=2, label='training error')

    pl.legend()
    pl.xlabel('Training Size')
    pl.ylabel('Error')

    pl.show()


def random_forest_curve(trees, X_train, y_train, X_test, y_test):
    """Calculate the model's performance improvement as the number of trees increases"""
    sizes = np.round(np.linspace(1, len(X_train), 50))
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))
    
    print trees
    
    for i, s in enumerate(sizes):
        # test the complexity, with different number of trees
        rf_clf = RandomForestRegressor(n_estimators=trees)
        # parameters n_estimators: number of trees in the forest
        # max_features: sqrt(n_features), n_features: number of features in data
        rf_clf.fit(X_train[:s], y_train[:s])
        train_err[i] = performance_metric(y_train[:s], rf_clf.predict(X_train[:s]))
        test_err[i] = performance_metric(y_test, rf_clf.predict(X_test))
        
    random_forest_curve_graph(sizes, train_err, test_err, trees)
    
    return test_err[-1]

    
def random_forest_curve_graph(sizes, train_err, test_err,trees):
    pl.figure()
    pl.title("Random Forest Learning Curve with {} Trees".format(trees))
    
    pl.plot(sizes, train_err, lw=2, label="Training Error")
    pl.plot(sizes, test_err, lw=2, label="Test Error")
    
    pl.xlabel("Training Size")
    pl.ylabel("Error")
    pl.show()
    


def model_complexity(X_train, y_train, X_test, y_test):
    """Calculate the performance of the model as model complexity increases."""

    print "Model Complexity: "

    max_depth = np.arange(1, 25)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        regressor = DecisionTreeRegressor(max_depth=d)
        # testing the performance, as complexity increases, using the same training data
        regressor.fit(X_train, y_train)

        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    model_complexity_graph(max_depth, train_err, test_err)
    
    return test_err[-1]


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
    
    
def boosting_optimization(X_train, y_train, X_test, y_test):
    
    gbm = GradientBoostingRegressor(n_estimators=3000, max_depth=10)
    gbm.fit(X_train, y_train)
    
    pred = gbm.predict(X_test)
    
    print "feature importances: "
    print pd.Series(gbm.feature_importances_, index=datasets.load_boston().feature_names)
    print "staged predict: {}".format(gbm.staged_predict(X_train))
    print "predict: {}".format(gbm.predict(X_test))
    print y_test
    
    #plot_boosting(pred)
    

def plot_boosting(pred):
    
    pl.figure()
    pl.title("Gradient Boosting")
    
    for i in np.nditer(pred):
        pl.plot(i, pred[i], lw=2, label="predictions")
    
    pl.legend()
    pl.xlabel("0")
    pl.ylabel("Error")
    
    pl.show()

# TODO: experiment and compare with xgboost

def fit_predict_model(city_data, x):
    """Find and tune the optimal model. Make a prediction on housing data."""
    
    X, y = city_data.data, city_data.target

    # Grid Search Parameters
    regressor = DecisionTreeRegressor()
    parameters = {'max_depth': range(1, 11)}    
    acc_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    
    # Run classifier to find the best param combination
    reg = GridSearchCV(regressor, parameters, scoring=acc_scorer)
    reg.fit(X, y)

    print "Best estimator: "
    print reg.best_estimator_
    print "Final Model: "
    print reg.fit(X, y)

    y = reg.predict(x)
    
    print "House: " + str(x)
    print "Prediction: " + str(y)
    
    
def find_three_neighbor_average(x):
    """Use a the nearest neighbor algorithm to verify the decision tree prediction"""
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

    city_data = datasets.load_boston()

    explore_city_data(city_data)

    X_train, y_train, X_test, y_test = split_data(city_data)
    
    results = {'tree': [], 'forest': [], 'boost': []}
    
    max_depths = range(1, 11)
    for max_depth in max_depths:
        results['tree'].append(learning_curve(max_depth, X_train, y_train, X_test, y_test))
        

    model_complexity(X_train, y_train, X_test, y_test)
    
    print "Starting Random Forest"
    a = time.time()
    forest = range(10, 21)
    for tree in forest:
        results['forest'].append(random_forest_curve(tree, X_train, y_train, X_test, y_test))
        
    print time.time() - a
        
    x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
    
    fit_predict_model(city_data, x)
    
    find_three_neighbor_average(x)
    
    results['boost'].append(boosting_optimization(X_train, y_train, X_test, y_test))
    
    print "results:\n"
    print results


if __name__ == "__main__":
    main()