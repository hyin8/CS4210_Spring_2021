#-------------------------------------------------------------------------
# AUTHOR: Haowen Yin
# FILENAME: naive_bayes.py
# SPECIFICATION: Implement naive bayes algorithm
# FOR: CS 4200- Assignment #2
# TIME SPENT: 40 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#reading the training data
#--> add your Python code here
db = []

with open('weather_training.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

# Dictionary Section
Day = {
    
}
Outlook = {
  'Sunny': 1,
  'Overcast': 2,
  'Rain': 3,
}
Temperature = {
  'Hot': 1,
  'Mild':2,
  'Cool': 3
}
Humidity = {
  'High': 1,
  'Normal': 2,
}
Wind={
  'Strong': 1,
  'Weak': 2
}
PlayTennis = {
    1: 'Yes',
    2: 'No'
}

attribute_dict = [Day,Outlook,Temperature,Humidity,Wind]

#transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X = []
for instance in db:
    temp = []
    for value in range(1,5):
        temp.append(attribute_dict[value][instance[value]])
    X.append(temp)
    temp.clear

#transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []
for instance in db:
    if instance[5] == 'Yes':
        Y.append(1)
    elif instance[5] == 'No':
        Y.append(2)

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the data in a csv file
#--> add your Python code here
db_test = []

with open('weather_test.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db_test.append (row)

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions.
#--> add your Python code here
#-->predicted = clf.predict_proba([[3, 1, 2, 1]])[0]
for i, instance in enumerate(db_test):
    X = []
    for value in range(1,5):
        X.append(attribute_dict[value][instance[value]])
    predicted_proba = clf.predict_proba([X])[0]
    predicted_class  = clf.predict([X])[0]

    if(predicted_proba[predicted_class-1]>= 0.75):
      print("D{day}".format(day = 15+i).ljust(15) + instance[1].ljust(15) + instance[2].ljust(15) + instance[3].ljust(15) + instance[4].ljust(15) + PlayTennis[predicted_class].ljust(15) + '{:.2f}'.format(predicted_proba[predicted_class-1]).ljust(15))
    

