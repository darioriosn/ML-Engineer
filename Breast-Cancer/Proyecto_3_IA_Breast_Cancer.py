#!/usr/bin/env python
# coding: utf-8

# Import packages

# # **Breast Cancer Analisys**
# 
# Based on the Breast Cancer Dataset of M YASSER H ("https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset")
# 

# ## **imports**

# In[1]:


import numpy as np # for linear algebra
import pandas as pd # data processing, CSV file I/O, etc
import seaborn as sns # for plots
import plotly.graph_objects as go # for plots
import plotly.express as px #for plots
import matplotlib.pyplot as plt # for visualizations and plots

from sklearn.model_selection import train_test_split # spliting training and testing data
from sklearn.preprocessing import MinMaxScaler, StandardScaler # data normalization with sklearn
from sklearn.linear_model import LogisticRegression, LinearRegression # model
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, confusion_matrix, ConfusionMatrixDisplay, recall_score # to evaluate the model
from mlxtend.plotting import plot_confusion_matrix # plot confusion matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # model
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import plot_tree, export_text, DecisionTreeClassifier
import missingno as msno # for plotting missing data

#undersampling
from imblearn.under_sampling import NearMiss
from collections import Counter

#oversampling
from imblearn.over_sampling import SMOTE, SVMSMOTE

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, cross_validate # Tuning


# ## **Read dataset**
# We have to read the CSV file, in this case it's *breast-cancer*
# 

# In[26]:


data = pd.read_csv('breast-cancer.csv')


# ##**Analyze dataset**
# We have to analyze the information from the dataset.
# If it contains null or empty data, incorrect data and so on.

# In[27]:


data.head()


# In[28]:


data.tail()


# ###**The Breast Cancer Dataset consists of 569 data points, with 32 features each**

# In[29]:


data.shape


# In[30]:


data.dtypes


# In[31]:


data.info()


# In[32]:


data.isnull().values.any()


# In[33]:


data.describe()


# In[34]:


data['diagnosis'].value_counts()


# ## **Plots**
# Let's make some charts/plots, to analyze graphically the dataset

# In[35]:


plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
sns.countplot(x='diagnosis', data=data, palette='Set2')
plt.ylabel('Diagnosis amount')
plt.xlabel('M = Malignant | B = Benign')


# In[36]:


plt.figure(figsize=(8,8))
pie = data['diagnosis'].value_counts()
explode = (0.05, 0)
colors = ['moccasin', 'coral']
labels = ['Benign', 'Malignant']
sns.set(font_scale=1.5)
plt.pie(pie, labels = labels, autopct = "%.2f%%", explode = explode, colors = colors)
plt.ylabel(labels, loc='bottom')


# In[37]:


px.pie(data, names='diagnosis')


# In[38]:


data.hist(figsize= (30,30), grid= True)


# ## **Clean Dataset**
# Clean Diagnosis Column
# and Drop ID Column

# In[39]:


columns_to_drop = ['id']
data = data.drop(columns=columns_to_drop)

data_clean = data
data_clean = data_clean.copy(deep = True)

data_clean.loc[data_clean['diagnosis'] == 'M', "diagnosis"] = 1
data_clean.loc[data_clean['diagnosis'] == 'B', "diagnosis"] = 0

data_clean['diagnosis'] = data_clean['diagnosis'].astype(int)


# In[40]:


data_clean['diagnosis'].value_counts()


# In[41]:


data_clean.info()


# In[42]:


print(data_clean.groupby('diagnosis').size())


# In[43]:


plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
sns.countplot(data_clean['diagnosis'])


# ### **Correlation matrix**

# First, let's create a copy of the dataset to clean up some data

# In[44]:


plt.figure(figsize=(40,40))
sns.set(font_scale=1)
sns.heatmap(data_clean.corr(numeric_only=True), annot=True, cmap='coolwarm', vmin=1.0, vmax=1.0, linecolor='white', linewidths=1).set_title("Correlation Matrix")


# In[45]:


data_clean.isnull().sum()/len(data_clean)*100


# ## **Outliers**

# In[46]:


data_clean.skew()


# In[47]:


Q1 = data_clean.quantile(.25)
Q3 = data_clean.quantile(.75)
IQR = Q1-Q3
IQR


# In[48]:


data_clean_out = data_clean[~((data_clean < (Q1-1.5 * IQR)) |  (data_clean > (Q3-1.5 * IQR))).any(axis=1)]
data_clean_out.shape


# ## Normalize and Balance the data (undersampling)
# ### Undersampling to avoid overfitting

# In[49]:


X = data_clean.copy(deep = True)

columns_to_drop = ['diagnosis']
X = X.drop(columns=columns_to_drop)
y = data_clean['diagnosis']


# In[50]:


##Undersampling
undersampling = NearMiss(version=1, n_neighbors=3)

X_resampled, y_resampled = undersampling.fit_resample(X,y)

data_balanced = pd.concat([X_resampled, y_resampled], axis=1)
data_balanced


# In[51]:


print(data_balanced.groupby('diagnosis').size())


# In[52]:


X = data_balanced[data_balanced.columns[:-1]]
y = data_balanced['diagnosis']


# In[53]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


# ### Scaling the data

# In[54]:


norm = MinMaxScaler().fit(X_train)
X_train_norm = norm.transform(X_train)
X_test_norm = norm.transform(X_test)


# ## **Data modeling**

# ![recall.png](data:image/png;base64,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)

# **Let's focus on the RECALL value**
# 
# 
# 
# The above equation can be explained by saying, from all the positive classes, how many we predicted correctly. Recall should be high as possible.
# 
# Recall calculates how many of the Actual Positives our model capture through labeling it as Positive (True Positive).
# We know that Recall shall be the model metric we use to select our best model when there is a high cost associated with False Negative.
# 
# "If a sick patient (Actual Positive) goes through the test and predicted as not sick (Predicted Negative). The cost associated with False Negative will be extremely high if the sickness is contagious."
# 
# In our case (Breast Cancer) it would be critical if a patient goes through a cancer test and the diagnosis is predicted as a Benign cancer but it actually is a Malignant cancer
# 
# Taken from:
# 
# https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
# 
# 
# https://towardsdatascience.com/understanding-confusion-matrix-a9ad42dcfd62

# ### **LogisticRegression**

# In[55]:


log_model = LogisticRegression()
log_model.fit(X_train_norm, y_train)
log_pred = log_model.predict(X_test_norm)

print("\n\t\t\t0 = Benign \n\t\t\t1 = Malignant")
print("\n", classification_report(y_test, log_pred))


# In[56]:


print("LogisticRegression Recall: {:.4f}".format(recall_score(y_test, log_pred)))

train_recall = recall_score(y_train, log_model.predict(X_train_norm))
print("Recall on training set: {:.2f}".format(train_recall))

test_recall = recall_score(y_test, log_model.predict(X_test_norm))
print("Recall on test set: {:.2f}".format(test_recall))


# In[57]:


labels = ['Benign', 'Malignant']
cm = confusion_matrix(y_test, log_pred)
plt.figure()
plot_confusion_matrix(cm, hide_ticks=True, cmap='Reds')
plt.xticks(range(2), labels, fontsize=10)
plt.yticks(range(2), labels, fontsize=10)


# #### HeatMap instead of plot_confusion_matrix

# In[58]:


ylabel = ["Actual [Benign]","Actual [Malignant]"]
xlabel = ["Pred [Benign]","Pred [Malignant]"]
plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, xticklabels = xlabel, yticklabels = ylabel, linecolor='white', linewidths=1)


# ### **RandomForestClassifier**

# In[59]:


rf_model = RandomForestClassifier()
rf_model.fit(X_train_norm, y_train)
rf_pred = rf_model.predict(X_test_norm)

print("\n\t\t\t0 = Benign \n\t\t\t1 = Malignant")
print("\n", classification_report(y_test, rf_pred))


# In[60]:


print("RandomForestClassifier Recall: {:.4f}".format(recall_score(y_test, rf_pred)))

train_recall = recall_score(y_train, rf_model.predict(X_train_norm))
print("Recall on training set: {:.2f}".format(train_recall))

test_recall = recall_score(y_test, rf_model.predict(X_test_norm))
print("Recall on test set: {:.2f}".format(test_recall))


# In[61]:


cm = confusion_matrix(y_test, rf_pred)
plt.figure()
plot_confusion_matrix(cm, hide_ticks=True, cmap='Reds')
plt.xticks(range(2), labels, fontsize=10)
plt.yticks(range(2), labels, fontsize=10)


# #### **Random Forest with params**

# In[62]:


rf1_model = RandomForestClassifier(max_depth=3, n_estimators=100, random_state=0)
rf1_model.fit(X_train_norm, y_train)
rf1_pred = rf1_model.predict(X_test_norm)

print("\t\t\t0 = Benign \n\t\t\t1 = Malignant")
print("\n", classification_report(y_train, rf1_model.predict(X_train_norm)))
print("\n", classification_report(y_test, rf1_model.predict(X_test_norm)))


# In[63]:


print("RandomForestClassifier 1 Recall: {:.4f}".format(recall_score(y_test, rf1_pred)))

train_recall = recall_score(y_train, rf1_model.predict(X_train_norm))
print("Recall on training set: {:.2f}".format(train_recall))

test_recall = recall_score(y_test, rf1_model.predict(X_test_norm))
print("Recall on test set: {:.2f}".format(test_recall))


# In[64]:


cm = confusion_matrix(y_test, rf1_pred)
plt.figure()
plot_confusion_matrix(cm, hide_ticks=True, cmap='Reds')
plt.xticks(range(2), labels, fontsize=10)
plt.yticks(range(2), labels, fontsize=10)


# ### **Support Vector Classification**

# In[65]:


svm_model = SVC().fit(X_train_norm, y_train)
svm_pred = svm_model.predict(X_test_norm)

print("\n\t\t\t0 = Benign \n\t\t\t1 = Malignant")
print("\n", classification_report(y_test, svm_pred))


# In[66]:


print("Support Vector Classification Recall: {:.4f}".format(recall_score(y_test, svm_pred)))

train_recall = recall_score(y_train, svm_model.predict(X_train_norm))
print("Recall on training set: {:.2f}".format(train_recall))

test_recall = recall_score(y_test, svm_model.predict(X_test_norm))
print("Recall on test set: {:.2f}".format(test_recall))


# In[67]:


cm = confusion_matrix(y_test, svm_pred)
plt.figure()
plot_confusion_matrix(cm, hide_ticks=True, cmap='Reds')
plt.xticks(range(2), labels, fontsize=10)
plt.yticks(range(2), labels, fontsize=10)


# #### **USING sklearn.metrics.ConfusionMatrixDisplay**

# In[68]:


svm_model = SVC().fit(X_train_norm, y_train)
svm_pred = svm_model.predict(X_test_norm)
disp = ConfusionMatrixDisplay.from_predictions(y_test, svm_pred , display_labels=labels)
disp.ax_.set_title("Support Vector Classification")
plt.show()


# ### **Gaussian NB**

# In[69]:


nb_model = GaussianNB()
nb_model.fit(X_train_norm, y_train)
nb_pred = nb_model.predict(X_test_norm)

print("\n\t\t\t0 = Benign \n\t\t\t1 = Malignant")
print("\n", classification_report(y_test, nb_pred))


# In[70]:


print("GaussianNB Recall: {:.4f}".format(recall_score(y_test, nb_pred)))

train_recall = recall_score(y_train, nb_model.predict(X_train_norm))
print("Recall on training set: {:.2f}".format(train_recall))

test_recall = recall_score(y_test, nb_model.predict(X_test_norm))
print("Recall on test set: {:.2f}".format(test_recall))


# In[71]:


cm = confusion_matrix(y_test, nb_pred)
plt.figure()
plot_confusion_matrix(cm, hide_ticks=True, cmap='Reds')
plt.xticks(range(2), labels, fontsize=10)
plt.yticks(range(2), labels, fontsize=10)


# ### **Decision Tree Classifier**

# In[72]:


dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_norm, y_train)
dt_pred = dt_model.predict(X_test_norm)

print("\n\t\t\t0 = Benign \n\t\t\t1 = Malignant")
print("\n", classification_report(y_test, dt_pred))


# In[73]:


print("DecisionTreeClassifier Recall: {:.4f}".format(recall_score(y_test, dt_pred)))

train_recall = recall_score(y_train, dt_model.predict(X_train_norm))
print("Recall on training set: {:.2f}".format(train_recall))

test_recall = recall_score(y_test, dt_model.predict(X_test_norm))
print("Recall on test set: {:.2f}".format(test_recall))


# In[74]:


cm = confusion_matrix(y_test, dt_pred)
plt.figure()
plot_confusion_matrix(cm, hide_ticks=True, cmap='Blues')
plt.xticks(range(2), labels, fontsize=10)
plt.yticks(range(2), labels, fontsize=10)


# In[75]:


plt.figure(figsize =(80,20))

plot_tree(dt_model, feature_names=X_train.columns, max_depth=4, filled=True);

X_train.columns


# ### **Decision Tree with Max Depth**

# In[76]:


tree_model = DecisionTreeClassifier(max_depth=3, random_state=0)
tree_model.fit(X_train_norm, y_train)
tree_pred = tree_model.predict(X_test_norm)

print("\n\t\t\t0 = Benign \n\t\t\t1 = Malignant")
print("\n", classification_report(y_test, tree_pred))


# In[77]:


print("DecisionTreeClassifier with Max Depth Recall: {:.4f}".format(recall_score(y_test, tree_pred)))

train_recall = recall_score(y_train, tree_model.predict(X_train_norm))
print("Recall on training set: {:.2f}".format(train_recall))

test_recall = recall_score(y_test, tree_model.predict(X_test_norm))
print("Recall on test set: {:.2f}".format(test_recall))


# In[78]:


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

acc_train_array = []
acc_valid_array = []
random_forest_settings = range(1, 21)


for max_d in random_forest_settings:
  tree = DecisionTreeClassifier(max_depth=max_d, random_state=42)
  tree.fit(X_train_norm, y_train)
  tree_pred = tree.predict(X_test_norm)

  y_pred = tree.predict(X_train_norm)
  acc_train = recall_score(y_train, y_pred)
  acc_train_array.append(acc_train)

  y_pred = tree.predict(X_test_norm)
  acc_valid = recall_score(y_test, y_pred)
  acc_valid_array.append(acc_valid)
  print("Recall score {:.4f}".format(acc_valid))


# In[79]:


print('For train set:')
print(acc_train_array)
print('For valid set:')
print(acc_valid_array)


plt.plot(random_forest_settings, acc_train_array, label="training accuracy")
plt.plot(random_forest_settings, acc_valid_array, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Random Forest")
plt.legend()
plt.savefig('random_forest_resolve_overfitting_model')


# ### **K_Neigbors**

# In[80]:


training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 4)

for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_norm, y_train)

    train_recall = recall_score(y_train, knn.predict(X_train_norm))
    test_recall = recall_score(y_test, knn.predict(X_test_norm))

    training_accuracy.append(train_recall)
    test_accuracy.append(test_recall)

plt.plot(neighbors_settings, training_accuracy, label="training recall")
plt.plot(neighbors_settings, test_accuracy, label="test recall")
plt.ylabel("Recall")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig('knn_compare_model')


# In[81]:


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_norm, y_train)
knn_pred = knn_model.predict(X_test_norm)

print("\n\t\t\t0 = Benign \n\t\t\t1 = Malignant")
print("\n", classification_report(y_test, knn_pred))


# In[82]:


print("KNeighborsClassifier Recall: {:.4f}".format(recall_score(y_test, knn_pred)))

train_recall = recall_score(y_train, knn_model.predict(X_train_norm))
print("Recall on training set: {:.2f}".format(train_recall))

test_recall = recall_score(y_test, knn_model.predict(X_test_norm))
print("Recall on test set: {:.2f}".format(test_recall))


# ### **Gradient Boosting**

# In[83]:


gb_model = GradientBoostingClassifier(random_state=0)
gb_model.fit(X_train_norm, y_train)
gb_pred = gb_model.predict(X_test_norm)

print("\n\t\t\t0 = Benign \n\t\t\t1 = Malignant")
print("\n", classification_report(y_test, gb_pred))


# In[84]:


print("Gradient Boosting Recall: {:.4f}".format(recall_score(y_test, gb_pred)))

train_recall = recall_score(y_train, gb_model.predict(X_train_norm))
print("Recall on training set: {:.2f}".format(train_recall))

test_recall = recall_score(y_test, gb_model.predict(X_test_norm))
print("Recall on test set: {:.2f}".format(test_recall))


# #### **Gradient Boosting with params**

# In[85]:


gb1 = GradientBoostingClassifier(random_state=0, max_depth=1)
gb1.fit(X_train_norm, y_train)

gb1_pred = gb1.predict(X_test_norm)

print("\n\t\t\t0 = Benign \n\t\t\t1 = Malignant")
print("\n", classification_report(y_test, gb1_pred))


# In[86]:


print("Gradient Boosting 1 Recall: {:.4f}".format(recall_score(y_test, gb1_pred)))

train_recall = recall_score(y_train, gb1.predict(X_train_norm))
print("Recall on training set: {:.2f}".format(train_recall))

test_recall = recall_score(y_test, gb1.predict(X_test_norm))
print("Recall on test set: {:.2f}".format(test_recall))


# In[87]:


gb2 = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gb2.fit(X_train_norm, y_train)
gb2_pred = gb2.predict(X_test_norm)

print("\n\t\t\t0 = Benign \n\t\t\t1 = Malignant")
print("\n", classification_report(y_test, gb2_pred))


# In[88]:


print("Gradient Boosting 2 Recall: {:.4f}".format(recall_score(y_test, gb2_pred)))

train_recall = recall_score(y_train, gb2.predict(X_train_norm))
print("Recall on training set: {:.2f}".format(train_recall))

test_recall = recall_score(y_test, gb2.predict(X_test_norm))
print("Recall on test set: {:.2f}".format(test_recall))


# **As seen above, the "max_depth" and the "learning_rate" did not help to get better results actually both GB1 and GB2 are worst than the GB model**

# ### **Neural Networks**

# In[89]:


mlp_model = MLPClassifier(random_state=42, max_iter=600)
mlp_model.fit(X_train_norm, y_train)
mlp_pred = mlp_model.predict(X_test_norm)

print("\n\t\t\t0 = Benign \n\t\t\t1 = Malignant")
print("\n", classification_report(y_test, mlp_pred))


# In[90]:


print("Neural Networks Recall: {:.4f}".format(recall_score(y_test, mlp_pred)))

train_recall = recall_score(y_train, mlp_model.predict(X_train_norm))
print("Recall on training set: {:.2f}".format(train_recall))

test_recall = recall_score(y_test, mlp_model.predict(X_test_norm))
print("Recall on test set: {:.2f}".format(test_recall))


# #### **Neural Network without the normalize data**

# In[91]:


mlp1 = MLPClassifier(random_state=42, max_iter=600)
mlp1.fit(X_train, y_train)
mlp1_pred = mlp1.predict(X_test_norm)

print("\n\t\t\t0 = Benign \n\t\t\t1 = Malignant")
print("\n", classification_report(y_test, mlp1_pred))


# In[92]:


print("Neural Networks 1 Recall: {:.4f}".format(recall_score(y_test, mlp1_pred)))

train_recall = recall_score(y_train, mlp1.predict(X_train))
print("Recall on training set: {:.2f}".format(train_recall))

test_recall = recall_score(y_test, mlp1.predict(X_test))
print("Recall on test set: {:.2f}".format(test_recall))


# **As seen above, the recall score on both training and test are 1 on the Neural Network without the normalized data, this means that the data has to be normalized, because if it is not normalized the model is overfitted**

# #### **Neural Network using Standard Scaler**

# In[93]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

mlp_standard = MLPClassifier(random_state=0, max_iter=300)
mlp_standard.fit(X_train_scaled, y_train)
mlp_standarized_pred = mlp_standard.predict(X_test_scaled)

print("\n\t\t\t0 = Benign \n\t\t\t1 = Malignant")
print("\n", classification_report(y_test, mlp_standarized_pred))


# In[94]:


print("Neural Networks Scaled Recall: {:.4f}".format(recall_score(y_test, mlp_standarized_pred)))

train_recall = recall_score(y_train, mlp_standard.predict(X_train_scaled))
print("Recall on training set: {:.2f}".format(train_recall))

test_recall = recall_score(y_test, mlp_standard.predict(X_test_scaled))
print("Recall on test set: {:.2f}".format(test_recall))


# **As seen above, also the recall score on both training and test are 1 on the Neural Network using Standard Scale, this means that the data has to be normalized using another algorithm, because if it is not normalized the model is overfitted**

# ### K-FOLD
# 

# In[95]:


models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))


names = []
scores = []
for name, model in models:
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_test_norm)
    scores.append(recall_score(y_test, y_pred))
    names.append(name)
tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)


# In[96]:


names = []
scores = []
for name, model in models:

    kfold = KFold(n_splits=10)
    score = cross_val_score(model, X_train_norm, y_train, cv=kfold, scoring='recall').mean()
    #score = cross_validate(model, X_train_norm, y_train, cv=kfold, scoring='recall')

    names.append(name)
    scores.append(score)
    print("Name: {}, Score: {:.4f}".format(name,score))


# ## **Hyperparameters**

# ### GridSearchCV and LogisticRegression

# In[97]:


c_values = list(np.arange(1, 10))
param_grid = [
    {'C': c_values, 'penalty': ['l1'], 'solver' : ['liblinear'], 'multi_class' : ['ovr']},
    {'C': c_values, 'penalty': ['l2'], 'solver' : ['liblinear', 'newton-cg', 'lbfgs'], 'multi_class' : ['ovr']}
]

grid = GridSearchCV(LogisticRegression(), param_grid, cv=kfold, scoring='recall')
grid.fit(X_train_norm, y_train)


# In[98]:


print(grid.best_params_)
print(grid.best_estimator_)


# In[99]:


log_with_params = LogisticRegression(C=6, multi_class='ovr', penalty='l1', solver='liblinear')
log_with_params_score = cross_val_score(log_with_params, X_train_norm, y_train, cv=kfold, scoring='recall').mean()
print("LogisticRegression's Recall using hyperparameters: {:.4f} ".format(log_with_params_score))


# ### Random Forest Turing with RandomSearch

# In[100]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]


# In[101]:


#Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[102]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf_random_search_model = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random_search = RandomizedSearchCV(estimator = rf_random_search_model, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random_search.fit(X_train_norm, y_train)


# In[103]:


print(rf_random_search.best_params_)
print(rf_random_search.best_estimator_)
print(rf_random_search.best_score_)


# In[104]:


train_recall = recall_score(y_train, rf_random_search.predict(X_train_norm))
print("Recall on training set: {:.2f}".format(train_recall))

test_recall = recall_score(y_test, rf_random_search.predict(X_test_norm))
print("Recall on test set: {:.2f}".format(test_recall))


# ## **Decision Tree Tuning with GridSearch**

# In[105]:


dt_gs_model = DecisionTreeClassifier(random_state=42)


# In[106]:


params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50],
    'criterion': ["gini", "entropy"]
}


# In[107]:


dt_grid_search = GridSearchCV(estimator=dt_gs_model,
                           param_grid=params,
                           cv=4, n_jobs=-1, verbose=1, scoring = "recall")


# In[108]:


dt_grid_search.fit(X_train_norm, y_train)


# In[109]:


print(dt_grid_search.best_params_)
print(dt_grid_search.best_estimator_)
print(dt_grid_search.best_score_)


# In[110]:


modelsBest = []
modelsBest.append(('LR', log_with_params))
modelsBest.append(('RF', rf_random_search))
modelsBest.append(('DT', dt_grid_search))


# In[111]:


names = []
scores = []
for name, model in modelsBest:
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_test_norm)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)
    names.append(name)
    print("Name: {}, Score: {:.4f}".format(name,score))
tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)


# In[112]:


axis = sns.barplot(x = 'Name', y = 'Score', data = tr_split)
axis.set(xlabel='Classifier', ylabel='Accuracy')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center")

plt.show()


# # **Analysis**

# 
