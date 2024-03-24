import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning

# Filter warnings for when model fails to fully converge to prevent console spam
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Number to be used for random operations to get reproducible results
random_seed = 42

def load_and_preprocess_data():

    # Get NBA Rookie Stats data to predict if a player will last over 5 years or not
    nba_rookie_stats_data = pd.read_csv('nba.csv')

    # Replace NaN values in each column with that columns average mean
    nba_rookie_stats_data.fillna(nba_rookie_stats_data.select_dtypes(include=['number']).mean(), inplace=True)

    return nba_rookie_stats_data

def predict_using_knn(X_training_features, X_testing_features, y_training_labels, y_testing_labels):

    # Create KNN classifier instance
    knn_classifier = KNeighborsClassifier()
    
    # Create param grid for KNN  i.e the hyperparameters we want to optimize
    # In this case, we want to optimize the number of neighors looked at to make a prediction
    knn_param_grid = {
        'n_neighbors': range(1, 30)
    }

    # Create and configure our Grid Search instance, using ten fold cross validation
    knn_grid_search = GridSearchCV(estimator=knn_classifier, param_grid=knn_param_grid, scoring='f1', cv=10)

    # Find the best hyperparameters for the knn model
    knn_grid_search.fit(X_training_features.to_numpy(), y_training_labels)

    # Get the best model found, and the best k value
    best_k_neighors = knn_grid_search.best_estimator_
    
    # Incase needed
    best_num_neighors = knn_grid_search.best_params_['n_neighbors']

    print(f"Best k hyperparameter for Best KNN Classifier without scaling using cross-validation is: {best_num_neighors}")

    # cross_val_score simply returns the f1 score of each fold as a numpy array.
    # Calculate average f1 score across all folds for best KNN model by taking mean of the array
    best_knn_average_f1_score = cross_val_score(best_k_neighors, X_training_features.to_numpy(), y_training_labels, cv=10, scoring='f1').mean()


    print(f"Average f1 score for Best KNN Classifier without scaling using cross-validation is: {best_knn_average_f1_score}")

    knn_y_testing_labels_predictions = best_k_neighors.predict(X_testing_features.to_numpy())

    best_knn_final_f1_score = f1_score(y_testing_labels, knn_y_testing_labels_predictions)

    print(f"Final f1 score for Best KNN Classifier without scaling on test dataset is: {best_knn_final_f1_score}")

    # Create a pipeline for the KNN classifier which will apply scaling when appropiate
    # Sklearn's standard scaler class is used
    # After data is scaled it is passed to the knn classifer to make predictions
    knn_with_scaling_pipeline = Pipeline([('scaler', StandardScaler()),
                             ('knn', KNeighborsClassifier())
                             ])
    
    # Create param grid for KNN pipeline, i.e the hyperparameters we want to optimize
    # In this case, we want to optimize the number of neighors looked at to make a prediction
    # We need to specify hyperparameters for a specific step in the pipeline, so we need to prefix the parameter 
    # with the step followed by __ 
    # Note: In this case the prefix is knn__ followed by n_neighors
    knn_with_scaling_param_grid = {
        'knn__n_neighbors': range(1, 30)
    }

    # Create and configure our Grid Search instance, using ten fold cross validation
    knn_with_scaling_grid_search = GridSearchCV(estimator=knn_with_scaling_pipeline, param_grid=knn_with_scaling_param_grid, scoring='f1', cv=10)

    # Find the best hyperparameters for the knn model
    knn_with_scaling_grid_search.fit(X_training_features, y_training_labels)

    # Get the best model found, and the best k values
    best_k_neighors_with_scaling = knn_with_scaling_grid_search.best_estimator_
    
    best_num_neighors_with_scaling = knn_with_scaling_grid_search.best_params_['knn__n_neighbors']

    print(f"Best k hyperparameter for Best KNN Classifier with scaling using cross-validation is: {best_num_neighors_with_scaling}")

    # cross_val_score simply returns the f1 score of each fold as a numpy array.
    # Calculate average f1 score across all folds for best KNN model by taking mean of the array
    best_knn_with_scaling_average_f1_score = cross_val_score(best_k_neighors_with_scaling, X_training_features, y_training_labels, cv=10, scoring='f1').mean()

    print(f"Average f1 score for Best KNN Classifier with scaling using cross-validation is: {best_knn_with_scaling_average_f1_score}")

    knn_with_scaling_y_testing_labels_predictions = best_k_neighors_with_scaling.predict(X_testing_features)

    best_knn_with_scaling_final_f1_score = f1_score(y_testing_labels, knn_with_scaling_y_testing_labels_predictions)

    print(f"Final f1 score for Best KNN Classifier with scaling on test dataset is: {best_knn_with_scaling_final_f1_score}")


# Predict whether a player will last more than five years using random forest
def predict_using_rf(X_training_features, X_testing_features, y_training_labels, y_testing_labels):
    
    # Create RF classifier instance
    rf_classifier = RandomForestClassifier(random_state=random_seed)
    
    # Create param grid for RF i.e the hyperparameters we want to optimize
    # n_estimators is the number of trees in the forest
    # max_depth is how far trees can grow until the leaf nodes are pure, or a node doesn't meet the min # of samples to split on
    # max features is the max number of features to be used by a decision tree when looking to make the best split on its nodes
    # Note: I made the choice to remain with the default sklearn gini criteria for splitting nodes due to the higher f1 scores 
    # that I recieved using gini index in the last assignment and in the interest of performance.
    # Initially I had more hyperparameters included in the GridSearch, however I opted to remove them in the interest of performance and the 
    # project requirements to use grid search. These were whether to use bootstrap, min samples for leaf node, and min samples to split
    rf_param_grid = {
        'n_estimators': [100, 500],
        'max_depth': [None, 5, 20],
        'max_features': [None, 'sqrt', 'log2'],
    }

    # Create and configure our Grid Search instance, using ten fold cross validation
    rf_grid_search = GridSearchCV(estimator=rf_classifier, param_grid=rf_param_grid, scoring='f1', cv=10)

    # Find the best hyperparameters for the knn model
    rf_grid_search.fit(X_training_features.to_numpy(), y_training_labels)

    # Get the best model found, and the best k value
    best_rf = rf_grid_search.best_estimator_
    
    best_num_trees = rf_grid_search.best_params_['n_estimators']
    best_max_depth = rf_grid_search.best_params_['max_depth']
    best_max_features = rf_grid_search.best_params_['max_features']

    print(f"Hyperparameters found for Best RF Classifier without scaling using cross-validation are the following:")
    print(f"Best Number of Trees: {best_num_trees}")
    print(f"Best Max Depth: {best_max_depth}")
    print(f"Best Max Number of Features per Subset: {best_max_features}")

    # cross_val_score simply returns the f1 score of each fold as a numpy array.
    # Calculate average f1 score across all folds for best rf model by taking mean of the array
    best_rf_average_f1_score = cross_val_score(best_rf, X_training_features.to_numpy(), y_training_labels, cv=10, scoring='f1').mean()

    print(f"Average f1 score for Best RF Classifier without scaling using cross-validation is: {best_rf_average_f1_score}")

    rf_y_testing_labels_predictions = best_rf.predict(X_testing_features.to_numpy())

    best_rf_final_f1_score = f1_score(y_testing_labels, rf_y_testing_labels_predictions)

    print(f"Final f1 score for Best RF Classifier without scaling on test dataset is: {best_rf_final_f1_score}")

    # Create a pipeline for the RF classifier which will apply scaling when appropiate
    # Sklearn's standard scaler class is used
    # After data is scaled it is passed to the RF classifer to make predictions
    rf_with_scaling_pipeline = Pipeline([('scaler', StandardScaler()),
                             ('rf', RandomForestClassifier(random_state=random_seed))
                             ])
    
    # Create param grid for RF i.e the hyperparameters we want to optimize
    # n_estimators is the number of trees in the forest
    # max_depth is how far trees can grow until the leaf nodes are pure, or a node doesn't meet the min # of samples to split ond
    # max features is the max number of features to be used by a decision tree when looking to make the best split on its nodes
    # This a param grid for a pipeline, so we must prefix the params with the name of the pipeline step they are for and __, so "rf__" is the prefix
    # I made the choice to remain with the default sklearn gini criteria for splitting nodes due to the higher f1 scores I recieved using gini index in the last assignment.
    # Initially I had more hyperparameters included in the GridSearch, however I opted to remove them in the interest of perforamnce and the 
    # project requirements to use grid search.
    rf_with_scaling_param_grid = {
    'rf__n_estimators': [100, 500],
    'rf__max_depth': [None, 5, 20],
    'rf__max_features': [None, 'sqrt', 'log2'],
    }

    # Create and configure our Grid Search instance, using ten fold cross validation
    rf_with_scaling_grid_search = GridSearchCV(estimator=rf_with_scaling_pipeline, param_grid=rf_with_scaling_param_grid, scoring='f1', cv=10)

    # Find the best hyperparameters for the random forest model
    rf_with_scaling_grid_search.fit(X_training_features, y_training_labels)

    # Get the best model found
    best_rf_with_scaling = rf_with_scaling_grid_search.best_estimator_
    
    # Get the best hyperparameters that were used by this model
    best_num_trees_with_scaling = rf_with_scaling_grid_search.best_params_['rf__n_estimators']
    best_max_depth_with_scaling = rf_with_scaling_grid_search.best_params_['rf__max_depth']
    best_max_features_with_scaling = rf_with_scaling_grid_search.best_params_['rf__max_features']

    print(f"Hyperparameters found for Best RF Classifier with scaling using cross-validation are the following:")
    print(f"Best Number of Trees: {best_num_trees_with_scaling}")
    print(f"Best Max Depth: {best_max_depth_with_scaling}")
    print(f"Best Max Number of Features per Subset: {best_max_features_with_scaling}")

    # cross_val_score simply returns the f1 score of each fold as a numpy array.
    # Calculate average f1 score across all folds for best RF model by taking mean of the array
    best_rf_with_scaling_average_f1_score = cross_val_score(best_rf_with_scaling, X_training_features, y_training_labels, cv=10, scoring='f1').mean()

    print(f"Average f1 score for Best RF Classifier with scaling using cross-validation is: {best_rf_with_scaling_average_f1_score}")

    rf_with_scaling_y_testing_labels_predictions = best_rf_with_scaling.predict(X_testing_features)

    best_rf_with_scaling_final_f1_score = f1_score(y_testing_labels, rf_with_scaling_y_testing_labels_predictions)

    print(f"Final f1 score for Best RF Classifier with scaling on test dataset is: {best_rf_with_scaling_final_f1_score}")

# Predict whether a player will last more than five years using logistic regression
def predict_using_lr(X_training_features, X_testing_features, y_training_labels, y_testing_labels):
    
    # Suppress warnings specific to lr
    warnings.filterwarnings("ignore", category=UserWarning, message="l1_ratio parameter is only used when penalty is 'elasticnet'")
    warnings.filterwarnings("ignore", category=FutureWarning, message="`penalty='none'`has been deprecated in 1.2 and will be removed in 1.4")
    warnings.filterwarnings("ignore", category=UserWarning, message="Setting penalty=None will ignore the C and l1_ratio parameters")

    # Create logistic regression classifier instance
    logistic_regression_classifier = LogisticRegression(random_state=random_seed)
    
    # Create param grid for logistic regression i.e the hyperparameters we want to optimize
    # In this case, those parameters are the regularization strength, and the penalty parameter.
    # We choose to use the saga solver as it the only one compatible with all supported penalty parameters.
    # The different penalty parameters are elasticnet, l1, l2, and none.
    # The regularization strength is given by C, where a smaller value implies stronger regularization.
    # l1_ratio is only used when penalty is elasticnet, essentially the ratio between the l1 and l2 penalty
    logistic_regression_param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['none', 'l1', 'l2', 'elasticnet'],
        'solver': ['saga'],
        'l1_ratio': [0.5]
    }

    # Create and configure our Grid Search instance, using ten fold cross validation
    logistic_regression_grid_search = GridSearchCV(estimator=logistic_regression_classifier, param_grid=logistic_regression_param_grid, scoring='f1', cv=10)

    # Find the best hyperparameters for the logistic regression model
    logistic_regression_grid_search.fit(X_training_features.to_numpy(), y_training_labels)

    # Get the best model found
    best_logistic_regression = logistic_regression_grid_search.best_estimator_
    
    # Get the best hyperparameters that were used for this model
    best_C = logistic_regression_grid_search.best_params_['C']
    best_penalty_parameter = logistic_regression_grid_search.best_params_['penalty']

    print(f"Hyperparameters found for Best Logistic Regression Classifier without scaling using cross-validation are the following:")
    print(f"Best C: {best_C}")
    print(f"Best Penalty Parameter: {best_penalty_parameter}")

    # cross_val_score simply returns the f1 score of each fold as a numpy array.
    # Calculate average f1 score across all folds for best logistic regression model by taking mean of the array
    best_logistic_regression_average_f1_score = cross_val_score(best_logistic_regression, X_training_features.to_numpy(), y_training_labels, cv=10, scoring='f1').mean()

    print(f"Average f1 score for Best Logistic Regression Classifier without scaling using cross-validation is: {best_logistic_regression_average_f1_score}")

    logistic_regression_y_testing_labels_predictions = best_logistic_regression.predict(X_testing_features.to_numpy())

    best_logistic_regression_final_f1_score = f1_score(y_testing_labels, logistic_regression_y_testing_labels_predictions)

    print(f"Final f1 score for Best Logistic Regression Classifier without scaling on test dataset is: {best_logistic_regression_final_f1_score}")

    # Create a pipeline for the Logistic Regression classifier which will apply scaling when appropiate
    # Sklearn's standard scaler class is used
    # After data is scaled it is passed to the RF classifer to make predictions
    logistic_regression_with_scaling_pipeline = Pipeline([('scaler', StandardScaler()),
                             ('lr', LogisticRegression(random_state=random_seed))
                             ])
    
    # Create param grid for logistic regression i.e the hyperparameters we want to optimize
    # In this case, those parameters are the regularization strength, and the penalty parameter.
    # We choose to use the saga solver as it the only one compatible with all supported penalty parameters.
    # The different penalty parameters are elasticnet, l1, l2, and none.
    # The regularization strength is given by C, where a smaller value implies stronger regularization.
    # Uses same param grid as without scaling, but with prefixes to ensure fairness and that the impact of scaling
    # is the only key thing being measured.
    logistic_regression_with_scaling_param_grid = {
        'lr__C': [0.1, 1, 10],
        'lr__penalty': ['none', 'l1', 'l2', 'elasticnet'],
        'lr__solver': ['saga'],
        'lr__l1_ratio': [0.5]
    }

    # Create and configure our Grid Search instance, using ten fold cross validation
    logistic_regression_with_scaling_grid_search = GridSearchCV(estimator=logistic_regression_with_scaling_pipeline, param_grid=logistic_regression_with_scaling_param_grid, scoring='f1', cv=10)

    # Find the best hyperparameters for the logistic regression model with scaling
    logistic_regression_with_scaling_grid_search.fit(X_training_features, y_training_labels)

    # Get the best model found
    best_logistic_regression_with_scaling = logistic_regression_with_scaling_grid_search.best_estimator_
    
    # Get the best hyperparameters that were used for this model
    best_C_with_scaling = logistic_regression_with_scaling_grid_search.best_params_['lr__C']
    best_penalty_parameter_with_scaling = logistic_regression_with_scaling_grid_search.best_params_['lr__penalty']

    print(f"Hyperparameters found for Best Logistic Regression Classifier with scaling using cross-validation are the following:")
    print(f"Best C: {best_C_with_scaling}")
    print(f"Best Penalty Parameter: {best_penalty_parameter_with_scaling}")


    # cross_val_score simply returns the f1 score of each fold as a numpy array.
    # Calculate average f1 score across all folds for best Logistic Regression model with scaling by taking mean of the array
    best_logistic_regression_with_scaling_average_f1_score = cross_val_score(best_logistic_regression_with_scaling, X_training_features, y_training_labels, cv=10, scoring='f1').mean()

    print(f"Average f1 score for Best Logistic Regression Classifier with scaling using cross-validation is: {best_logistic_regression_with_scaling_average_f1_score}")

    logistic_regression_with_scaling_y_testing_labels_predictions = best_logistic_regression_with_scaling.predict(X_testing_features)

    best_logistic_regression_with_scaling_final_f1_score = f1_score(y_testing_labels, logistic_regression_with_scaling_y_testing_labels_predictions)

    print(f"Final f1 score for Best Logistic Regression Classifier with scaling on test dataset is: {best_logistic_regression_with_scaling_final_f1_score}")


# Predict whether a player will last more than five years using a Multi-layer Perceptron classifier model
def predict_using_mlp(X_training_features, X_testing_features, y_training_labels, y_testing_labels):
    
    # Create MLP classifier instance
    mlp_classifier = MLPClassifier(random_state=random_seed)

    # Create param grid for MLP Classifier i.e the hyperparameters we want to optimize
    # In this case, those parameters are the learning rate (convergence speed), the regularization strength (prevent overfitting),
    # the number of hidden layers, and the number of neurons per each of those layer. In addition we experiment with the solver used, 
    # sklearn documentation recommends adam for relatively large datasets and lbfgs for small datasets. 
    # Finally, we also test two different activation functions, relu and the logistic sigmoid function.
    mlp_classifier_param_grid = {
        'learning_rate_init': [0.001, 0.01, 0.1], 
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'hidden_layer_sizes': [(100,),(50, 50), (100, 50, 25)],
        'solver': ['adam', 'lbfgs'],
        'activation': ['relu', 'logistic']
    }

    # Create and configure our Grid Search instance, using ten fold cross validation
    mlp_classifier_grid_search = GridSearchCV(estimator=mlp_classifier, param_grid=mlp_classifier_param_grid, scoring='f1', cv=10)

    # Find the best hyperparameters for the mlp classifier model
    mlp_classifier_grid_search.fit(X_training_features.to_numpy(), y_training_labels)

    # Get the best model found
    best_mlp_classifier = mlp_classifier_grid_search.best_estimator_
    
    # Get the best hyperparameters that were used for this model
    best_learning_rate_init = mlp_classifier_grid_search.best_params_['learning_rate_init']
    best_alpha = mlp_classifier_grid_search.best_params_['alpha']
    best_hidden_layer_sizes = mlp_classifier_grid_search.best_params_['hidden_layer_sizes']
    best_solver = mlp_classifier_grid_search.best_params_['solver']
    best_activation = mlp_classifier_grid_search.best_params_['activation']

    print(f"Hyperparameters found for Best Multi-layer Perceptron Classifier without scaling using cross-validation are the following:")
    print(f"Best Learning Rate: {best_learning_rate_init}")
    print(f"Best Alpha (Regularization Strength): {best_alpha}")
    print(f"Best Hidden Layer Sizes ( Note: # Elements = # Num Hidden Layers. Each element in the tuple represents the number of neurons in a layer): {best_hidden_layer_sizes}")
    print(f"Best solver algorithm is {best_solver}")
    print(f"Best activation function is {best_activation}")

    # cross_val_score simply returns the f1 score of each fold as a numpy array.
    # Calculate average f1 score across all folds for best mlp classifier model by taking mean of the array
    best_mlp_classifier_average_f1_score = cross_val_score(best_mlp_classifier, X_training_features.to_numpy(), y_training_labels, cv=10, scoring='f1').mean()

    print(f"Average f1 score for Best Multi-layer Perceptron Classifier without scaling using cross-validation is: {best_mlp_classifier_average_f1_score}")

    mlp_classifier_y_testing_labels_predictions = best_mlp_classifier.predict(X_testing_features.to_numpy())

    best_mlp_classifier_final_f1_score = f1_score(y_testing_labels, mlp_classifier_y_testing_labels_predictions)

    print(f"Final f1 score for Best MLP Classifier without scaling on test dataset is: {best_mlp_classifier_final_f1_score}")

    # Create a pipeline for the MLP Perceptron classifier which will apply scaling when appropiate
    # Sklearn's standard scaler class is used
    # After data is scaled it is passed to the MLP Perceptron classifer to make predictions
    mlp_classifier_with_scaling_pipeline = Pipeline([('scaler', StandardScaler()),
                             ('mlp', MLPClassifier(random_state=random_seed))
                             ])
    
    # Create param grid for MLP Classifier i.e the hyperparameters we want to optimize
    # In this case, those parameters are the learning rate (convergence speed), the regularization strength (prevent overfitting),
    # the number of hidden layers, and the number of neurons per each of those layers. In addition we experiment with the solver used, 
    # sklearn documentation recommends adam for relatively large datasets and lbfgs for small datasets. 
    # Finally, we also test two different activation functions, relu and the logistic sigmoid function.
    # Same param grid is used as with out scaling, only with different param names (pipeline step prefixes added) to ensure fairness.
    mlp_classifier_with_scaling_param_grid = {
        'mlp__learning_rate_init': [0.001, 0.01, 0.1], 
        'mlp__alpha': [0.0001, 0.001, 0.01, 0.1],
        'mlp__hidden_layer_sizes': [(100,),(50, 50), (100, 50, 25)],
        'mlp__solver': ['adam', 'lbfgs'],
        'mlp__activation': ['relu', 'logistic'],
    }

    # Create and configure our Grid Search instance, using ten fold cross validation
    mlp_classifier_with_scaling_grid_search = GridSearchCV(estimator=mlp_classifier_with_scaling_pipeline, param_grid=mlp_classifier_with_scaling_param_grid, scoring='f1', cv=10)

    # Find the best hyperparameters for the MLP classifier model with scaling
    mlp_classifier_with_scaling_grid_search.fit(X_training_features, y_training_labels)

    # Get the best model found
    best_mlp_classifier_with_scaling = mlp_classifier_with_scaling_grid_search.best_estimator_
    
    # Get the best hyperparameters that were used for this model
    best_learning_rate_init_with_scaling = mlp_classifier_with_scaling_grid_search.best_params_['mlp__learning_rate_init']
    best_alpha_with_scaling = mlp_classifier_with_scaling_grid_search.best_params_['mlp__alpha']
    best_hidden_layer_sizes_with_scaling = mlp_classifier_with_scaling_grid_search.best_params_['mlp__hidden_layer_sizes']
    best_solver_with_scaling = mlp_classifier_with_scaling_grid_search.best_params_['mlp__solver']
    best_activation_with_scaling = mlp_classifier_with_scaling_grid_search.best_params_['mlp__activation']

    print(f"Hyperparameters found for Best Multi-layer Perceptron Classifier with scaling using cross-validation are the following:")
    print(f"Best Learning Rate: {best_learning_rate_init_with_scaling}")
    print(f"Best Alpha (Regularization Strength): {best_alpha_with_scaling}")
    print(f"Best Hidden Layer Sizes ( Note: # Elements = # Num Hidden Layers. Each element in the tuple represents the number of neurons in a layer): {best_hidden_layer_sizes_with_scaling}")
    print(f"Best solver algorithm is {best_solver_with_scaling}")
    print(f"Best activation function is {best_activation_with_scaling}")

    # cross_val_score simply returns the f1 score of each fold as a numpy array.
    # Calculate average f1 score across all folds for best multi-layer perceptron classifier model with scaling by taking mean of the array
    best_mlp_classifier_with_scaling_average_f1_score = cross_val_score(best_mlp_classifier_with_scaling, X_training_features, y_training_labels, cv=10, scoring='f1').mean()

    print(f"Average f1 score for Best Multi-layer Perceptron Classifier with scaling using cross-validation is: {best_mlp_classifier_with_scaling_average_f1_score}")

    mlp_classifier_with_scaling_y_testing_labels_predictions = best_mlp_classifier_with_scaling.predict(X_testing_features)

    best_mlp_classifier_with_scaling_final_f1_score = f1_score(y_testing_labels, mlp_classifier_with_scaling_y_testing_labels_predictions)

    print(f"Final f1 score for Best Multi-layer Perceptron Classifier with scaling on test dataset is: {best_mlp_classifier_with_scaling_final_f1_score}")

def main(algorithm_arguments):

    cleaned_dataset = load_and_preprocess_data()
    
    # Create a new dataframe from the original, dropping the target column to get our training features.
    # Note : We drop name since it is non numeric and most algorithms we use in this project expect numeric data. 
    X_features = cleaned_dataset.drop(['Name', 'TAR'], axis=1)

    # Get a series containing all rows with only the last column which represents the target
    y_target = cleaned_dataset['TAR']

    # Randomly split the original data into training set and testing sets, where 80% of the examples are in the training set, and the other 20% in the testing sets
    X_training_features, X_testing_features, y_training_labels, y_testing_labels = train_test_split(X_features, y_target, test_size=0.20, random_state=random_seed)

    '''
        # For debugging

        print("X_train shape:", X_training_features.shape) 
        print("X_test shape:", X_testing_features.shape)
        print("y_train shape:", y_training_labels.shape)
        print("y_test shape:", y_testing_labels.shape)

        print(cleaned_dataset.isnull().values.any())
        print(cleaned_dataset.isnull().sum())
    '''

    # We only run the algorithms for which user provided input to run this algorithm

    if algorithm_arguments.knn:
        # Predict using K Nearest Neighbors and print results
        predict_using_knn(X_training_features=X_training_features, X_testing_features=X_testing_features, y_training_labels=y_training_labels, y_testing_labels=y_testing_labels)
    
    if algorithm_arguments.rf:
        # Predict using Random Forest and print results
        predict_using_rf(X_training_features=X_training_features, X_testing_features=X_testing_features, y_training_labels=y_training_labels, y_testing_labels=y_testing_labels)
  
    if algorithm_arguments.lr:
        # Predict using Logistic Regression and print results
        predict_using_lr(X_training_features=X_training_features, X_testing_features=X_testing_features, y_training_labels=y_training_labels, y_testing_labels=y_testing_labels)

    if algorithm_arguments.mlp:
        # Predict using Multi-layer Perception Classifier
        predict_using_mlp(X_training_features=X_training_features, X_testing_features=X_testing_features, y_training_labels=y_training_labels, y_testing_labels=y_testing_labels)

if __name__ == "__main__":

    # Create argument parser class instance to add our arguments to 
    argument_parser = argparse.ArgumentParser(description="This is a script designed to predict whether or not an NBA player will last longer than five years based on their stats. ")

    # Define our arguments, if user doesn't provide one of them when running the script, it'll be set to False by default
    # Note: If an argument is true, the corresponding algorithm will be used when passed to main, otherwise, it will not be run.
    argument_parser.add_argument('--knn', action="store_true", help="Use K Nearest Neighbors to Predict if Player Duration >= 5 Years ")
    argument_parser.add_argument('--rf', action="store_true", help="Use Random Forest to Predict if Player Duration >= 5 Years ")
    argument_parser.add_argument('--lr', action="store_true", help="Use Logistic Regression to Predict if Player Duration >= 5 Years ")
    argument_parser.add_argument('--mlp', action="store_true", help="Use Multi-layer Perceptron Classifier to Predict if Player Duration >= 5 Years ")

    # When running the script, parse the user's arguments and store them into an arguments object which we pass to main
    # The argument object's attributes are True or False value which tell us whether or not to run an algorithm
    algorithm_arguments = argument_parser.parse_args()

    # Pass user's input to the main function so we can start processing the data and begin predictions
    main(algorithm_arguments)