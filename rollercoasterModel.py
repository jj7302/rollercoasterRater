from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

def get_data_from_csv(filename):
    import csv
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        return list(reader)

raw_data = get_data_from_csv('rollercoasters.csv') #opens the csv file and store it as a python variable

def rearrange_csv_data(dataList):
    #exclude the first row
    data = dataList[1:]
    features = []
    targets = []

    for row in data: #should really be called columns and not rows
        del row[0:4]
        del row[1]
        del row[2]
        del row[3]
        for i in range(len(row)):
            row[i] = float(row[i])
        features.append(row[3:])
        targets.append([row[0]]) #rows 1 and 2 at this point have intensity and sickness metrics, only selcting 0 for the target

    return np.array(features), np.array(targets)

features, targets = rearrange_csv_data(raw_data) #gets the columns we want from the csv file and
#a list of features in order: [max_speed,avg_speed,ride_time,ride_length,max_pos_gs,max_neg_gs,max_lateral_gs,total_air_time,drops,highest_drop_height,inversions]
#the target is the excitement column from the csv

xTrain, xTest, yTrain, yTest = train_test_split(features, targets, random_state=7) #

#all of the following models were tried and the one that preforms best, the Random Forest decision tree, was left uncommented
#model = KNeighborsRegressor(n_neighbors=2)
#model = LinearRegression()
#model = Ridge()
#model = SVR(C=10, gamma=.7)
#model = MLPRegressor(solver='lbfgs', random_state=2, hidden_layer_sizes=[100, 100])

model = RandomForestRegressor(n_estimators=500, random_state=9)
model = Pipeline([("scaler", MinMaxScaler()), ("decisionTree",  model)])

''' This code was used to select the best parameter for n_estimators, the result was frequently 500 so that will be used to train a model.
param_grid = {'decisionTree__n_estimators': [5, 10, 50, 100, 150, 250, 500]}
grid_search = GridSearchCV(model, param_grid, cv=5, return_train_score=True)
print(cross_val_score(model, feautures, targets).mean()) #cross-val scores were compared between all the model choices to select one
model = grid_search.fit(xTrain, yTrain)
print(model.best_params_)
print(model.score(xTest, yTest))
'''

model.fit(xTrain, yTrain)
predictions = model.predict(xTest)
actual = yTest

for prediction in range(len(predictions)):
    print('Predicted: ' + str(predictions[prediction]))
    print('Actual: ' + str(predictions[prediction]))
    print(' ')




print(model.score(xTrain, yTrain))
print(model.score(xTest, yTest))
print('')

#this function returns a thrillingness score for each rollercoaster based on the following rollercoaster features
def rate_rollercoaster(max_speed,avg_speed,ride_time,ride_length,max_pos_gs,max_neg_gs,max_lateral_gs,total_air_time,drops,highest_drop_height,inversions):
    data = [[max_speed,avg_speed,ride_time,ride_length,max_pos_gs,max_neg_gs,max_lateral_gs,total_air_time,drops,highest_drop_height,inversions]]
    return model.predict(data)

def rate_rollercoaster_from_list(data):
    return model.predict([data])

print(rate_rollercoaster(31,11,70,1591,3.13,-1.7,2.15,1.44,11,22,0))

