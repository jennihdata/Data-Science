import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, average_precision_score, plot_precision_recall_curve



import seaborn as sns
from sklearn import preprocessing
from yellowbrick.model_selection import FeatureImportances
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from yellowbrick.model_selection import FeatureImportances
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('US_Accidents_2019_cleaned.csv')

# Class Balancing | Using Up Sampling

# Separate majority and minority classes
severity_counts = df["Severity"].value_counts()

# plt.figure(figsize=(10, 8))
# plt.title("Histogram for the severity")
# sns.barplot(severity_counts.index, severity_counts.values)
# plt.xlabel("Severity")
# plt.ylabel("Value")
# plt.show()

df_s1 = df[df['Severity'] == 1]
df_s2 = df[df['Severity'] == 2]
df_s3 = df[df['Severity'] == 3]
df_s4 = df[df['Severity'] == 4]

count = max(df_s1.count()[0], df_s2.count()[0], df_s3.count()[0], df_s4.count()[0])

# Upsample minority class
df_s1 = resample(df_s1, replace=df_s1.count()[0] < count, n_samples=count, random_state=42)
df_s2 = resample(df_s2, replace=df_s2.count()[0] < count, n_samples=count, random_state=42)
df_s3 = resample(df_s3, replace=df_s3.count()[0] < count, n_samples=count, random_state=42)
df_s4 = resample(df_s4, replace=df_s4.count()[0] < count, n_samples=count, random_state=42)

# Combine majority class with upsampled minority class
df = pd.concat([df_s1, df_s2, df_s3, df_s4])

# Display new class counts
df.groupby(by='Severity')['Severity'].count()

# plt.figure(figsize=(10, 8))
# plt.title("Histogram for the severity")
# sns.barplot(df.index, df.values)
# plt.xlabel("Severity")
# plt.ylabel("Value")
# plt.show()


target='Severity'
features = ['Start_Lat','Start_Lng','Temperature(F)','Humidity(%)','Pressure(in)', 'Visibility(mi)','Wind_Speed(mph)','Hour','Duration']

# Create arrays for the features and the response variable

# set X and y
y = df[target]
X = df[features]

# Split the data set into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# List of classification algorithms
algo_lst = ['Decision Tree','Random Forest','Naive Bayes','Logistic Regression','SVM','MLP','KNN']

# Initialize an empty list for the accuracy for each algorithm
accuracy_lst = []


# DECISION TREE ALGORITHM
dt_entropy = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)
# Fit dt_entropy to the training set
dt_entropy.fit(X_train, y_train)
y_pred = dt_entropy.predict(X_test)
accuracy_entropy = accuracy_score(y_test, y_pred)
accuracy_lst.append(accuracy_entropy)
print('[Decision Tree -- entropy] accuracy_score: {:.3f}.'.format(accuracy_entropy))

# viz = FeatureImportances(dt_entropy)
# viz.fit(X,y)
# viz.show()

# print("Confusion Matrix\n -------------------")
# cm = confusion_matrix(y_test, y_pred)
#
# index = ["Actual Severity 1", "Actual Severity 2", "Actual Severity 3", "Actual Severity 4"]
# columns = ["Predicted Severity 1", "Predicted Severity 2", "Predicted Severity 3", "Predicted Severity 4"]
# conf_matrix = pd.DataFrame(data=cm, columns=columns, index=index)
# plt.figure(figsize=(8, 5))
# sns.heatmap(conf_matrix, annot=True, fmt=".3f",linewidths=.5, square = True, cmap = 'Blues_r')
# plt.title("Confusion Matrix - DECISION TREE")
# plt.show()
#
# print("\nClassification Report\n -----------------------")
# print(classification_report(y_test, y_pred))

# RANDOM FOREST ALGORITHM
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
accuracy_lst.append(acc)
print("[Random forest algorithm] accuracy_score: {:.3f}.".format(acc))

# viz = FeatureImportances(clf)
# viz.fit(X,y)
# viz.show()

# print("Confusion Matrix\n -------------------")
# cm = confusion_matrix(y_test, y_pred)
#
# index = ["Actual Severity 1", "Actual Severity 2", "Actual Severity 3", "Actual Severity 4"]
# columns = ["Predicted Severity 1", "Predicted Severity 2", "Predicted Severity 3", "Predicted Severity 4"]
# conf_matrix = pd.DataFrame(data=cm, columns=columns, index=index)
# plt.figure(figsize=(8, 5))
# sns.heatmap(conf_matrix, annot=True, fmt=".3f",linewidths=.5, square = True, cmap = 'Blues_r')
# plt.title("Confusion Matrix - RANDOM FOREST")
# plt.show()
#
# print("\nClassification Report\n -----------------------")
# print(classification_report(y_test, y_pred))


# NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
modelnb = GaussianNB()
nbtrain = modelnb.fit(X_train, y_train)
y_pred = nbtrain.predict(X_test)
acc=accuracy_score(y_test, y_pred)
accuracy_lst.append(acc)
print("[Naive Bayes algorithm] accuracy_score: {:.3f}.".format(acc))

# print("Confusion Matrix\n -------------------")
# cm = confusion_matrix(y_test, y_pred)
#
# index = ["Actual Severity 1", "Actual Severity 2", "Actual Severity 3", "Actual Severity 4"]
# columns = ["Predicted Severity 1", "Predicted Severity 2", "Predicted Severity 3", "Predicted Severity 4"]
# conf_matrix = pd.DataFrame(data=cm, columns=columns, index=index)
# plt.figure(figsize=(8, 5))
# sns.heatmap(conf_matrix, annot=True, fmt=".3f",linewidths=.5, square = True, cmap = 'Blues_r')
# plt.title("Confusion Matrix - NAIVE BAYES")
# plt.show()
#
# print("\nClassification Report\n -----------------------")
# print(classification_report(y_test, y_pred))


# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train_scaled, y_train)
y_pred = logisticRegr.predict(X_test_scaled)
acc=accuracy_score(y_test, y_pred)
accuracy_lst.append(acc)
print("[Logistic regression algorithm] accuracy_score: {:.3f}.".format(acc))

# viz = FeatureImportances(logisticRegr)
# viz.fit(X,y)
# viz.show()

# print("Confusion Matrix\n -------------------")
# cm = confusion_matrix(y_test, y_pred)
#
# index = ["Actual Severity 1", "Actual Severity 2", "Actual Severity 3", "Actual Severity 4"]
# columns = ["Predicted Severity 1", "Predicted Severity 2", "Predicted Severity 3", "Predicted Severity 4"]
# conf_matrix = pd.DataFrame(data=cm, columns=columns, index=index)
# plt.figure(figsize=(8, 5))
# sns.heatmap(conf_matrix, annot=True, fmt=".3f",linewidths=.5, square = True, cmap = 'Blues_r')
# plt.title("Confusion Matrix - Logistic Regression")
# plt.show()
#
# print("\nClassification Report\n -----------------------")
# print(classification_report(y_test, y_pred))



# SVM
clf = SVC(gamma='auto', kernel='rbf', random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
accuracy_lst.append(acc)
print("[Support Vector Machine algorithm] accuracy_score: {:.3f}.".format(acc))

# print("Confusion Matrix\n -------------------")
# cm = confusion_matrix(y_test, y_pred)
#
# index = ["Actual Severity 1", "Actual Severity 2", "Actual Severity 3", "Actual Severity 4"]
# columns = ["Predicted Severity 1", "Predicted Severity 2", "Predicted Severity 3", "Predicted Severity 4"]
# conf_matrix = pd.DataFrame(data=cm, columns=columns, index=index)
# plt.figure(figsize=(8, 5))
# sns.heatmap(conf_matrix, annot=True, fmt=".3f",linewidths=.5, square = True, cmap = 'Blues_r')
# plt.title("Confusion Matrix - SVM")
# plt.show()
#
# print("\nClassification Report\n -----------------------")
# print(classification_report(y_test, y_pred))


# MULTI LAYER PERCEPTON
mlp_clf = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1).fit(X_train_scaled, y_train)
y_pred = mlp_clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
accuracy_lst.append(acc)
print("[MLP algorithm] accuracy_score: {:.3f}.".format(acc))
print("MLP" ,mlp_clf.score(X_test_scaled, y_test))

# print("Confusion Matrix\n -------------------")
# cm = confusion_matrix(y_test, y_pred)
#
# index = ["Actual Severity 1", "Actual Severity 2", "Actual Severity 3", "Actual Severity 4"]
# columns = ["Predicted Severity 1", "Predicted Severity 2", "Predicted Severity 3", "Predicted Severity 4"]
# conf_matrix = pd.DataFrame(data=cm, columns=columns, index=index)
# plt.figure(figsize=(8, 5))
# sns.heatmap(conf_matrix, annot=True, fmt=".3f",linewidths=.5, square = True, cmap = 'Blues_r')
# plt.title("Confusion Matrix - MLP")
# plt.show()
#
# print("\nClassification Report\n -----------------------")
# print(classification_report(y_test, y_pred))


# KNN
knn = KNeighborsClassifier(n_neighbors = 15, weights='uniform')
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
accuracy_lst.append(acc)
print("[KNN 15 algorithm] accuracy_score: {:.3f}.".format(acc))

# print("Confusion Matrix\n -------------------")
# cm = confusion_matrix(y_test, y_pred)
#
# index = ["Actual Severity 1", "Actual Severity 2", "Actual Severity 3", "Actual Severity 4"]
# columns = ["Predicted Severity 1", "Predicted Severity 2", "Predicted Severity 3", "Predicted Severity 4"]
# conf_matrix = pd.DataFrame(data=cm, columns=columns, index=index)
# plt.figure(figsize=(8, 5))
# sns.heatmap(conf_matrix, annot=True, fmt=".3f",linewidths=.5, square = True, cmap = 'Blues_r')
# plt.title("Confusion Matrix - Logistic KNN")
# plt.show()
#
# print("\nClassification Report\n -----------------------")
# print(classification_report(y_test, y_pred))

# print("\nRoc Curve\n -------------------")
# import matplotlib.pyplot as plt
# from sklearn import datasets, metrics
# metrics.plot_roc_curve(knn, X_test, y_test)
# plt.show()


# Make a plot of the accuracy scores for different algorithms
# Generate a list of ticks for y-axis
y_ticks = np.arange(len(algo_lst))

# Combine the list of algorithms and list of accuracy scores into a dataframe, sort the value based on accuracy score
df_acc = pd.DataFrame(list(zip(algo_lst, accuracy_lst)), columns=['Algorithm','Accuracy_Score']).sort_values(by=['Accuracy_Score'],ascending = True)

# Make a plot
ax = df_acc.plot.barh('Algorithm', 'Accuracy_Score', align='center',legend=False,color='0.5')

# Add the data label on to the plot
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+0.02, i.get_y()+0.2, str(round(i.get_width(),2)), fontsize=10)

# Set the limit, lables, ticks and title
plt.xlim(0,1.1)
plt.xlabel('Accuracy Score')
plt.yticks(y_ticks, df_acc['Algorithm'], rotation=0)
plt.title('Which algorithm is better?')

plt.show()

