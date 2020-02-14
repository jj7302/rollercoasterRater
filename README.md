# Rollercoaster Model
The aim of this repository is to create a model to rate how thrilling rollercoasters are based on a couple of factors. rollercoasterModel.py contains python code which utilizes scikit learn to train a Random Forest Decision Tree regressor on the data in the csv. I tried a number of other models, including a Support Vector Regression and a deep neural network, but opted to use the decision tree for the final model because it yielded the best performance. The other models which were not used are commented out. A Grid Search was used to optimize the parameter number of estimators for the decision tree. The final result/ goal of the program is a function called rate_rollercoaster which takes as arguments all of the different attributes of the rollercoaster being considered in the model and gives as an output an excitment score. The original data is in rollercoasters.csv and was gotten from Nolan Conaway of the data science website kaggle: https://www.kaggle.com/nolanbconaway/rollercoaster-tycoon-rides/data. This data is not from actual rollecoasters but is from a game called roller coaster tycon, since this game provided excitment scores and clean data for the rollercoasters. I created this model as practice for an upcoming math modelling competition, so using data about real rollercoasters was not as much of a concern.


