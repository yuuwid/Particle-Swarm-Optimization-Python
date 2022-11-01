from pso.optimizer import PSO

from sklearn import metrics
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load Iris Dataset
iris = datasets.load_iris()

# Split Feature and Label
X, y = datasets.load_iris(return_X_y=True)

# Split Data Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


###################################
######     Random Forest     ######
###################################

# Create Model Random Forest
clf = RandomForestClassifier(n_estimators=100)
# Fit data to Model
clf.fit(X_train, y_train)
# Predict Data Test
y_pred_rf = clf.predict(X_test)

###################################
######  Random Forest + PSO  ######
###################################

# PSO params
c1 = 1.0
c2 = 2.0
w = 0.5
n_population = 20
max_iter = 70

# Create Model PSO
pso = PSO(c1, c2, w, n_population, max_iter, 'mse')

# Create Model Random Forest
model_pso = RandomForestClassifier()
# Fit Model Random Forest to PSO
pso.fit_model(model_pso)
# Fit Data Train and Test to PSO
pso.fit(X_train, X_test, y_train, y_test)
# Optimize
best_model_rf = pso.optimize()
# Predict Data Test
y_pred_rf_opt = best_model_rf.predict(X_test)


###################################
######       Accuracy        ######
###################################

print("RANDOM FOREST NORMAL IRIS DATASET")
print("ACCURACY:", round(metrics.accuracy_score(y_test, y_pred_rf) * 100, 4), "%")

print()

print("RANDOM FOREST + PSO IRIS DATASET")
print("ACCURACY:", round(metrics.accuracy_score(
    y_test, y_pred_rf_opt) * 100, 4), "%")
