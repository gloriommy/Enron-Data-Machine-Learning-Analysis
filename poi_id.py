#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier
from tester import dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
import math
import numpy

### Task 1: Select Features.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary', 'to_messages', 'deferral_payments', 
'exercised_stock_options', 'bonus', 'restricted_stock',
 'shared_receipt_with_poi', 'restricted_stock_deferred', 
 'total_stock_value', 'expenses', 'loan_advances', 'from_messages',
 'other', 'from_this_person_to_poi', 'director_fees',
 'deferred_income',  'long_term_incentive', 'from_poi_to_this_person', "to_poi/total_to_messages"] # You will need to use more features

 
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0) # removing the outlier. 

### Task 3: Creating a new feature (proportion of emails sent to persons of interest to total emails sent)
names = data_dict.keys()
for name in names:
    data_dict[name]["to_poi/total_to_messages"] = float(data_dict[name]["from_this_person_to_poi"]) / float(data_dict[name]["to_messages"])
    #if data_dict[name]["to_poi/total_to_messages"] == "nan":
    if math.isnan(data_dict[name]["to_poi/total_to_messages"]):
        data_dict[name]["to_poi/total_to_messages"] = 0
    
### Store to my_dataset for easy export below. 
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Trying a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
#clf = GaussianNB() #Accuracy: 0.37713       Precision: 0.15600      Recall: 0.83250
#clf = tree.DecisionTreeClassifier()  #Accuracy: 0.80233       Precision: 0.24751      Recall: 0.23650
#clf = GradientBoostingClassifier() #Accuracy: 0.85473       Precision: 0.41841      Recall: 0.22950
#clf = SVC()

# Final, best performing model
clf = SVC(kernel = "sigmoid", C = 45, gamma = 0.5)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


