import pandas as pd 
from itertools import combinations
from numpy import array,array_equal
import scipy as sp
from sklearn import cross_validation as cv
from sklearn import tree
from sklearn import metrics
from sklearn import ensemble
from sklearn import linear_model 
from sklearn import naive_bayes 

# Any results you write to the current directory are saved as output.
def print_shapes():
    print('Train: {}\nTest: {}'.format(train_dataset.shape, test_dataset.shape))
train_dataset = pd.read_csv("C:\\Users\\samee\\Desktop\\Git\\Santander\\train\\train.csv")
test_dataset = pd.read_csv('C:\\Users\\samee\\Desktop\\Git\\Santander\\test\\test.csv')
print_shapes()

# How many nulls are there in the datasets?
nulls_train = (train_dataset.isnull().sum()==1).sum()
nulls_test = (test_dataset.isnull().sum()==1).sum()

# Remove constant features
def identify_constant_features(dataframe):
    count_uniques = dataframe.apply(lambda x: len(x.unique()))
    constants = count_uniques[count_uniques == 1].index.tolist()
    return constants
constant_features_train = set(identify_constant_features(train_dataset))

# Drop the constant features
train_dataset.drop(constant_features_train, inplace=True, axis=1)
test_dataset.drop(constant_features_train, inplace=True, axis=1)
print_shapes()

# Remove equals features
def identify_equal_features(dataframe):
    features_to_compare = list(combinations(dataframe.columns.tolist(),2))
    equal_features = []
    for compare in features_to_compare:
        is_equal = array_equal(dataframe[compare[0]],dataframe[compare[1]])
        if is_equal:
            equal_features.append(list(compare))
    return equal_features
equal_features_train = identify_equal_features(train_dataset)

# Remove the second feature of each pair.
features_to_drop = array(equal_features_train)[:,1] 
train_dataset.drop(features_to_drop, axis=1, inplace=True)
test_dataset.drop(features_to_drop, axis=1, inplace=True)
print_shapes()

# Define the variables model.
y_name = 'TARGET'
feature_names = train_dataset.columns.tolist()
feature_names.remove(y_name)
X = train_dataset[feature_names]
y = train_dataset[y_name]

# Save the features selected for later use.
pd.Series(feature_names).to_csv('features_selected_step1.csv', index=False)
print('Features selected\n{}'.format(feature_names))
print(len(feature_names))
   
print_shapes()
# Proportion of classes
y.value_counts()/len(y)

skf = cv.StratifiedKFold(y, n_folds=3, shuffle=True)
score_metric = 'roc_auc'
scores = {}

def score_model(model):
    return cv.cross_val_score(model, X, y, cv=skf, scoring=score_metric)

# Determining the best model
scores['tree'] = score_model(tree.DecisionTreeClassifier())
scores['extra_tree'] = score_model(ensemble.ExtraTreesClassifier())
scores['forest'] = score_model(ensemble.RandomForestClassifier())
scores['ada_boost'] = score_model(ensemble.AdaBoostClassifier())
scores['bagging'] = score_model(ensemble.BaggingClassifier())
scores['grad_boost'] = score_model(ensemble.GradientBoostingClassifier())
scores['ridge'] = score_model(linear_model.RidgeClassifier())
scores['passive'] = score_model(linear_model.PassiveAggressiveClassifier())
scores['sgd'] = score_model(linear_model.SGDClassifier())
scores['gaussian'] = score_model(naive_bayes.GaussianNB())


# Print the scores
model_scores = pd.DataFrame(scores).mean()
model_scores.sort_values(ascending=False)
model_scores.to_csv('model_scores.csv', index=False)
print('Model scores\n{}'.format(model_scores))

# Test data on the best models
model1 = ensemble.AdaBoostClassifier()
model1.fit(X,y)
model1.score(X,y)
print_shapes()
predicted_output1=model1.predict_proba(test_dataset)
submission1 = pd.DataFrame({"ID":test_dataset.index,"TARGET":predicted_output1[:,1]})
submission1.to_csv("Submission1.csv",encoding='utf-8')

model2 = ensemble.GradientBoostingClassifier()
model2.fit(X,y)
model2.score(X,y)
print_shapes()
predicted_output2=model2.predict_proba(test_dataset)
submission2 = pd.DataFrame({"ID":test_dataset.index,"TARGET":predicted_output2[:,1]})
submission2.to_csv("Submission2.csv",encoding='utf-8')