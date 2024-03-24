# NBA Player Longevity Prediction

This sklearn project is intended to predict whether or not an NBA player will last five years or longer in the NBA given their rookie stats. It uses various well known machine learning classification algorithms with and without scaling, compares their average f1 scores using cross validation, and their f1 score on the final test dataset. GridSearch is also used for each model with and without normalization (scaling) in attempt to find the hyperparameters which provide the optimal f1 scores for each model when using scaling and when not given the dataset used. I created this repository for my CAP6610 - Machine Learning Class, project 2 submission at the University of North Florida, taught by Dr. Liu.

# Dataset

The NBA Player Rookie Stats dataset is stored in the `nba.csv` file in this repository. The features are the players stats during their rookie season, and the target variable is whether they lasted more than or equal to five years in the NBA.

## Prerequisites

In order to run my project, you need to have the following dependencies installed:

- Python version 3.x (Note: If you have issues with other versions, I developed and tested this script with 3.11.5)
- pandas
- scikit-learn
- argparse

You can install the required Python packages yourself using pip or simply use the Anaconda environment as I did:

To install with pip

```bash
pip install pandas scikit-learn argparse
```

## Usage

To run the script and make the predictions, use the following command

python main.py [--knn] [--rf] [--lr] [--mlp]

- `--knn`: Use K-Nearest Neighbors algorithm for prediction
- `--rf`: Use Random Forest algorithm for prediction
- `--lr`: Use Logistic Regression algorithm for prediction
- `--mlp`: Use Multi-layer Perceptron Classifier for prediction

You may choose to provide one or more of the arguments listed above in order to have more control of which algorithms are used to make a prediction. 

Note: If you do not provide any arguments, the data will be loaded and preprocessed but no actual predictions will be made.

Example Usage: 

```bash
python main.py --knn --rf
```

The above command will run predictions just for K-Nearest Neighbors and Random Forest. Logistic Regression and Multi-layer Perceptron Classifier will be skipped.

# Results

The following will be output for each model with and without scaling that you choose to run:

- Best hyperparameters found using GridSearch
- Average F1 score found through cross-validation
- Final F1 score of the algorithm model with the best hyperparameters on the test dataset

Results for a model will be displayed seperately with and without normalizing (feature scaling) to analyze its impact on the F1 score.

# Acknowledgements
- A great deal of the code was implemented by heavily referencing sklearn documentation and examples for the respective classes of the models implemented.
    - [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.kneighbors)
    - [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier)
    - [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)
    - [MLPClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- The [Mahesh Huddar YouTube channel](https://www.youtube.com/@MaheshHuddar) was very helpful in further understanding the different algorithms and how they work theoretically under the hood in scikit-learn.
- Special thanks to my Professor Dr. Liu for introducing us to these challenging ML concepts and giving us actionable projects to earn valuable machine learning experience and expand our coding portfolios.

Please feel to provide any suggestions or report any bugs you may find in order to help enhance this repository. 