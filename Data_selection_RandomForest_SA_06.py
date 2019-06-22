import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import pydot
from sklearn import preprocessing, model_selection
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from treeinterpreter import treeinterpreter as ti
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Preparation of initial dataset
df_name1 = 'Dataset_SA_06'

path = 'C:/Users/1pc/PycharmProjects/AI/Prediction models/ARTICLE/'
df_name2 = '.csv'
df_name3 = path + df_name1 + df_name2
datas = pd.read_csv(df_name3)
datas['date'] = pd.to_datetime(datas['time'], unit='s')
datas.drop(datas.tail(1).index,inplace=True)
datas['date'] = pd.to_datetime(datas['time'], unit='s')
list2 = datas.drop('time', axis=1)
date = list2['date']
list2.index = pd.MultiIndex.from_product([date])
list3 = list2.drop('date', axis=1)
forecast_col = 'Price_BTC'
price2 = list3
price2['label'] = list3[forecast_col]
df = price2
df.fillna(-99999, inplace=True)


# Technical indicators calculation on the dataset
close = df['Price_BTC']
open1 = df['Open_BTC']
low = df['Low_BTC']
high = df['High_BTC']
volume = df['Volume_BTC']

n = 7
n_slow = 14

mfm = ((close - low) - (high - close) / (high - low))
mfv = mfm * volume
adl = mfm.cumsum(axis=0)
df['ADL'] = adl
df['MA'] = pd.Series(close.rolling(n, min_periods=n).mean(), name='MA_' + str(12))
df['EMA'] = pd.Series(close.ewm(span=n, min_periods=n).mean(), name='EMA_' + str(n))
df['Average'] = (high+low)/2
df['Momentum'] = pd.Series(close.diff(n), name='Momentum_' + str(n))

M = close.diff(n - 1)
N = close.shift(n - 1)
df['ROC'] = pd.Series(M / N, name='ROC_' + str(n))

df['MSD'] = pd.Series(close.rolling(n, min_periods=n).std())
b1 = 4 * df['MSD'] / df['MA']
B1 = pd.Series(b1, name='Bollinger')
df = df.join(B1)
b2 = (close - df['MA'] + 2 * df['MSD']) / (4 * df['MSD'])
df['BB'] = pd.Series(b2, name='Bollinger%b_' + str(n))

df['SOk'] = pd.Series((close - low) / (high - low), name='SO%k')
df['SOd'] = pd.Series(df['SOk'].ewm(span=n, min_periods=n).mean(), name='SO%d_' + str(n))

df['EMAfast'] = pd.Series(close.ewm(span=n, min_periods=n_slow).mean())
df['EMAslow'] = pd.Series(close.ewm(span=n_slow, min_periods=n_slow).mean())
df['MACD'] = pd.Series(df['EMAfast'] - df['EMAslow'], name='MACD_' + str(n) + '_' + str(n_slow))
df['MACDsign'] = pd.Series(df['MACD'].ewm(span=n, min_periods=n).mean(), name='MACDsign_' + str(n) + '_' + str(n_slow))
df['MACDdiff'] = pd.Series(df['MACD'] - df['MACDsign'], name='MACDdiff_' + str(n) + '_' + str(n_slow))

ad = (2 * close - high - low) / (high - low) * volume
df['Chaikin'] = pd.Series(ad.ewm(span=3, min_periods=3).mean() - ad.ewm(span=10, min_periods=10).mean(), name='Chaikin')

# df['STD'] = pd.Series(close.rolling(n, min_periods=n).std(), name='STD_' + str(n))
df['Force_Index'] = pd.Series(close.diff(n) * volume.diff(n), name='Force_' + str(n))

# Activate to recieve SA #7
"""""
df = df.drop(columns=['Force_Index', 'Chaikin', 'MACDdiff',
                      'MACDsign', 'MACD', 'SOd', 'SOk', 'BB', 'Bollinger',
                      'ROC', 'Momentum', 'twitter_lists', 'twitter_following',
                      'twitter_followers', 'reddit_posts_per_day', 'reddit_comments_per_hour',
                      'reddit_comments_per_day', 'fb_talking_about', 'code_repo_closed_pull_issues',
                      'Volume_DASH', 'Volume_XRP', 'Price_XRP', 'Volume_ETH', 'Low_ETH', 'Volume_BTC',
                      'twitter_favourites', 'reddit_active_users', 'influence_page_views',
                      'code_repo_open_issues', 'Open_XRP', 'MSD', 'twitter_statuses', 'High_ETH',
                      'Low_XRP', 'High_XRP', 'Open_ETH', 'reddit_subscribers', 'Price_DASH',
                      'Price_ETH', 'High_DASH', 'Open_DASH', 'Low_DASH'])
"""""

# Number of days for training and testing datasets
n1 = 150
n2 = 1000 - n1
n3 = n2 - 1

# Dataset for preprocessing
df['label'] = df[forecast_col]
features = df.drop(columns='label')
X = np.array(df.drop(['label'], 1))
feature_list = list(features.columns)
n_feat = len(features.columns)
n_feat1 = int(n_feat * 0.18)

# Data preprocessing
X = preprocessing.scale(X)
X_lately = X[n2:]
X = X[0:n3]
df.dropna(inplace=True)
y = np.array(df['label'])
y1 = y[0:n3]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y1, test_size=0.2, random_state=42)


# Creating and training the model
clf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=2, max_features=n_feat1,
                            min_samples_leaf=1, max_leaf_nodes=6,
                            min_samples_split=100, min_weight_fraction_leaf=0.0,
                            n_estimators=100, n_jobs=1, oob_score=True, random_state=54,
                            verbose=0, warm_start=True)

fit1 = clf.fit(X_train, y_train)

# Preliminary accuracy calculation
accuracy = clf.score(X_train, y_train)
print('Train accuracy is %f' % accuracy)
accuracy2 = clf.score(X_test, y_test)
print('Test accuracy is %f' % accuracy2)

# Forecast on the testing dataset
forecast, bias, contributions = ti.predict(clf, X_lately)
forecast_set = [item for sublist in forecast for item in sublist]


# Cross-validation test for confidence
cvs = cross_val_score(fit1, X_train, y_train)
print(cvs)

# Metrics calculation
list4 = list3.tail(n1)
dates = list2['date'].tail(n1)
list4['Forecast'] = forecast_set
test = list4['Price_BTC']

rms=np.sqrt(np.mean(np.power((test-forecast_set),2)))
mape = np.mean(np.abs((test - forecast_set) / test)) * 100
mse = mean_squared_error(test, forecast_set)
mae = mean_absolute_error(test, forecast_set)
pcc = stats.pearsonr(forecast_set, test)

print(rms)
print(mape)
print(mse)
print(mae)
print(pcc)

# Performance metrics
errors = abs(forecast_set - test)
errors1 = round(np.mean(errors), 2)
print('Average absolute error:', errors1, 'USD.')
mape2 = 100 * (errors / test)
accuracy3 = 100 - np.mean(mape2)
acc3 = round(accuracy3, 2)
print('Accuracy:', acc3, '%.')

# Text file to write down metrics
text_file = open("Test_model_specification_SA_06.txt", "w")
text_file.write('RMS: ' +str(rms))
text_file.write(', MAPE: ' +str(mape))
text_file.write(', MSE: ' +str(mse))
text_file.write(', MAE: ' +str(mae))
text_file.write(', PCC: ' +str(pcc[0]))
text_file.write(', Average absolute error:' + str(errors1) + 'USD.')
text_file.write(', Accuracy:' + str(acc3) + '%.')
text_file.close()


# Get numerical feature importances
importances = list(clf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# Creating Figure to draw importance
fig= plt.figure(figsize=(16,10))
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')
plt.tight_layout()
# plt.rcParams.update({'figure.autolayout': True})
plt.rcParams['figure.figsize'] = (16.0, 16.0)
plt.savefig('fig_SA_06_importance.png')
plt.show()

# List of features sorted from most to least important
sorted_importances = [importance[1] for importance in feature_importances]
sorted_features = [importance[0] for importance in feature_importances]
# Cumulative importances
cumulative_importances = np.cumsum(sorted_importances)
# Make a line graph
plt.plot(x_values, cumulative_importances, 'g-')
# Draw line at 95% of importance retained
plt.hlines(y = 0.95, xmin=0, xmax=len(sorted_importances), color = 'r', linestyles = 'dashed')
# Format x ticks and labels
plt.xticks(x_values, sorted_features, rotation = 'vertical')
# Axis labels and title
plt.xlabel('Variable')
plt.ylabel('Cumulative Importance')
plt.title('Cumulative Importances')
plt.savefig('fig_SA_06_importance_cumulative.png')
plt.show()


# Tree sample creation. DOT and PNG files.
# Activate if needed
"""""
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
"""""
tree_small = clf.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'Tree_sample.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('Tree_sample.dot')
graph.write_png('Tree_sample.png')
