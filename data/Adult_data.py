
# Commented out IPython magic to ensure Python compatibility.
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %matplotlib inline

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
CURRENT_DIR = os.path.abspath(os.path.dirname(__name__))
DATA_DIR = os.path.join('')
TRAIN_DATA_FILE = os.path.join('./data/adult.data')
TEST_DATA_FILE = os.path.join('./data/adult.test')
from collections import OrderedDict

data_types = OrderedDict([
    ("age", "int"),
    ("workclass", "category"),
    ("final_weight", "int"),  # originally it was called fnlwgt
    ("education", "category"),
    ("education_num", "int"),
    ("marital_status", "category"),
    ("occupation", "category"),
    ("relationship", "category"),
    ("race", "category"),
    ("sex", "category"),
    ("capital_gain", "float"),  # required because of NaN values
    ("capital_loss", "int"),
    ("hours_per_week", "int"),
    ("native_country", "category"),
    ("income_class", "category"),
])
target_column = "income_class"

def read_dataset(path):
    return pd.read_csv(
        path,
        names=data_types,
        index_col=None,

        comment='|',  # test dataset has comment in it
        skipinitialspace=True,  # Skip spaces after delimiter
        na_values={
            'capital_gain': 99999,
            'workclass': '?',
            'native_country': '?',
            'occupation': '?',
        },
        dtype=data_types,
    )

def clean_dataset(data):
    # Test dataset has dot at the end, we remove it in order
    # to unify names between training and test datasets.
    data['income_class'] = data.income_class.str.rstrip('.').astype('category')
    
    # Remove final weight column since there is no use
    # for it during the classification.
    data = data.drop('final_weight', axis=1)
    
    # Duplicates might create biases during the analysis and
    # during prediction stage they might give over-optimistic
    # (or pessimistic) results.
    data = data.drop_duplicates()
    
    # Binarize target variable (>50K == 1 and <=50K == 0)
    data[target_column] = (data[target_column] == '>50K').astype(int)

    return data

def deduplicate(train_data, test_data):
    train_data['is_test'] = 0
    test_data['is_test'] = 1

    data = pd.concat([train_data, test_data])
    # For some reason concatenation converts this column to object
    data['native_country'] = data.native_country.astype('category')
    data = data.drop_duplicates()
    
    train_data = data[data.is_test == 0].drop('is_test', axis=1)
    test_data = data[data.is_test == 1].drop('is_test', axis=1)
    
    return train_data, test_data

#train_data = clean_dataset(read_dataset(TRAIN_DATA_FILE))
#test_data = clean_dataset(read_dataset(TEST_DATA_FILE))

# Note that we did de-duplication per dataset, but there are duplicates
# between training and test data. With duplicates between datasets
# we will might get overconfident results.
#train_data, test_data = deduplicate(train_data, test_data)
#print("Percent of the positive classes in the training data: {:.2%}".format(np.mean(train_data.income_class)))
def get_categorical_columns(data, cat_columns=None, fillna=True):
    if cat_columns is None:
        cat_data = data.select_dtypes('category')
    else:
        cat_data = data[cat_columns]

    if fillna:
        for colname, series in cat_data.iteritems():
            if 'Other' not in series.cat.categories:
                series = series.cat.add_categories(['Other'])

            cat_data[colname] = series.fillna('Other')
            
    return cat_data

def features_with_one_hot_encoded_categories(data, cat_columns=None, fillna=True):
    cat_data = get_categorical_columns(data, cat_columns, fillna)
    one_hot_data = pd.get_dummies(cat_data)
    df = pd.concat([data, one_hot_data], axis=1)

    features = [
        'age',
        'education_num',
        'hours_per_week',
        'capital_gain',
        'capital_loss',
    ] + one_hot_data.columns.tolist()
    features.remove('sex_Other')
    X = df[features].fillna(0).values.astype(float)
    y = df[target_column].values
    
    return X, y


#Generating Training and Test data. column 65 coresponds to white race and column 63 coresponds to black race. 
def Adult_dataset(seed = 0):
  from sklearn.model_selection import train_test_split
  train_data = clean_dataset(read_dataset(TRAIN_DATA_FILE))
  test_data = clean_dataset(read_dataset(TEST_DATA_FILE))
  train_data, test_data = deduplicate(train_data, test_data)
  X_train, y_train = features_with_one_hot_encoded_categories(train_data)
  X_test, y_test = features_with_one_hot_encoded_categories(test_data)

  X = np.concatenate((X_train, X_test), axis=0)
  Y = np.concatenate((y_train, y_test), axis=0).reshape(-1, )
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = seed)

  s = 1
  m = 0
  y_train_0 = y_train[(X_train[:,65]==1)]
  y_train_1 = y_train[(X_train[:,63]==1)]
  y_train = y_train[((X_train[:,65]==1) | (X_train[:,63]==1)) ]

  X_train_0 = (X_train[(X_train[:,65]==1) ,: ]-m)/s
  X_train_1 = (X_train[(X_train[:,63]==1),: ]-m)/s
  X_train = (X_train[((X_train[:,65]==1) | (X_train[:,63]==1)),: ]-m)/s

  n_0 = X_train_0.shape[0]
  n_1 = X_train_1.shape[0]
  d_in = X_train_0.shape[1]
  d_out = 1

  y_test_0 = y_test[(X_test[:,65]==1)]
  y_test_1 = y_test[(X_test[:,63]==1)]
  y_test = y_test[((X_test[:,65]==1) | (X_test[:,63]==1)) ]

  X_test_0 = (X_test[(X_test[:,65]==1) ,: ] - m)/s
  X_test_1 = (X_test[(X_test[:,63]==1),: ] - m)/s
  X_test = (X_test[((X_test[:,65]==1) | (X_test[:,63]==1)) , : ]-m)/s
  permutation = np.random.RandomState(seed).permutation(X_train.shape[0])
  np.take(X_train,permutation,axis=0,out=X_train) 
  np.take(y_train,permutation,axis=0,out=y_train) 
  permutation = np.random.RandomState(seed).permutation(X_test.shape[0])
  np.take(X_test,permutation,axis=0,out=X_test) 
  np.take(y_test,permutation,axis=0,out=y_test) 
  return X_train_0, y_train_0, X_train_1, y_train_1, X_test_0, y_test_0, X_test_1, y_test_1