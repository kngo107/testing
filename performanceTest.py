from auto_ml import Predictor
from auto_ml.utils import get_boston_dataset
import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_digits
from sklearn.datasets import load_linnerud
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer

from sklearn import metrics
from sklearn import model_selection
from sklearn import neural_network


def _testWithData(dataset):
    count = 1
    metrics = _getMetrics(dataset, 3)
    for metric, data in metrics.iteritems():
        plt.figure(count)
        count +=1
        _plotMetric(metric,data)

def _plotMetric(metric, data):

    #Set title and Label:
    plt.title("{} Score of per dataset".format(metric.upper()))
    plt.xlabel('Dataset')
    plt.ylabel(metric)

    bar_width = 0.35
    opacity = 0.8
    n_groups = 0
    auto_ml_score = []
    mlp_score = []
    x_label = []
    for dataset_name,scores in data.iteritems():
        n_groups +=1
        x_label.append(dataset_name)
        auto_ml_score.append(scores['auto_ml'])
        mlp_score.append(scores['mlp']) 
    index = np.arange(n_groups)
    rects1 = plt.bar(index, tuple(auto_ml_score), bar_width,
                    alpha=opacity,
                    color='k',
                    label='auto_ml')
    rects2 = plt.bar(index + bar_width, tuple(mlp_score), bar_width,
                    alpha=opacity,
                    color='r',
                    label='mlp')
    plt.xticks(index + bar_width, tuple(x_label))
    plt.legend()
    plt.tight_layout()
    plt.show()



def _getMetrics(dataset, n_fold):
    results = {'mse':dict(), 'r2':dict()}
    #Get metrics for all datasets
    while (dataset):
        dataset_name = dataset[0]
        if dataset[0] == 'boston':
            #SPECIFIC FIELDS============================================
            #regressor problem
            #506 instasnces, 13 attributes (numeric/categorical)
            problem_type = 'regressor'
            output = 'MEDV'
            column_descriptions = {
                    output: 'output',
                    'CHAS': 'categorical'
                    }
            raw_data = load_boston()
            #===========================================================
            df = pd.DataFrame(raw_data.data)
            df.columns = raw_data.feature_names
            df[output] = raw_data['target']
            train_set, test_set = model_selection.train_test_split(df, test_size=0.4, random_state=42)
            train_set_x = train_set.drop(output,axis=1)
        else:
            exit()


        if problem_type == 'regressor':
            result = {'mse':{dataset_name:{'auto_ml':0,'mlp':0}},
                      'r2':{dataset_name:{'auto_ml':0,'mlp':0}}}
        else:
            result = {}
        
        average_factor = 1.0/n_fold
        for i in range(n_fold):

            ml_predictor = Predictor(type_of_estimator=problem_type, column_descriptions=column_descriptions)
            ml_predictor.train(train_set, verbose=False, compare_all_models = True)

            y_auto_ml_predicted = ml_predictor.predict(test_set)


            if problem_type == 'regressor':
                train_set_y = np.asarray(train_set[output], dtype=np.float64)
                mlp = neural_network.MLPRegressor(hidden_layer_sizes=(13,13,13), max_iter = 1000)
                mlp.fit(train_set_x,train_set_y)
            else:
                train_set_y = np.asarray(train_set[output], dtype="|S6")
                mlp = neural_network.MLPClassifier(hidden_layer_sizes=(13,13,13), max_iter = 1000)
                mlp.fit(train_set_x,train_set_y)

            y_mlp_predicted = mlp.predict(test_set.drop(output,axis=1))

            if problem_type == 'regressor':
                result['mse'][dataset_name]['auto_ml'] += (metrics.mean_squared_error(test_set['MEDV'],y_auto_ml_predicted)*average_factor)
                result['r2'][dataset_name]['auto_ml'] += (metrics.r2_score(test_set['MEDV'],y_auto_ml_predicted)*average_factor)
                result['mse'][dataset_name]['mlp'] += (metrics.mean_squared_error(test_set['MEDV'], y_mlp_predicted)*average_factor)
                result['r2'][dataset_name]['mlp'] += (metrics.r2_score(test_set['MEDV'],y_mlp_predicted)*average_factor)



        #Take average
        for test, data in result.iteritems():
            results[test][dataset_name] = data[dataset_name]  


        del dataset[0]
    return results

_testWithData(['boston'])
