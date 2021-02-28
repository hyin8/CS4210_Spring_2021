#-------------------------------------------------------------------------
# AUTHOR: Haowen Yin
# FILENAME: knn.py
# SPECIFICATION: Implement 1NN algorithm
# FOR: CS 4200- Assignment #2
# TIME SPENT: 40 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from os import error
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []
error = 0
total = 0

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)


#loop your data to allow each instance to be your test set
for i, instance in enumerate(db):

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]
    #--> add your Python code here
    X = []
    for item in db:
        if item != instance:
            X.append([int(item[0]),int(item[1])])

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]
    #--> add your Python code here
    Y = []
    for item in db:
        if item != instance:
            Y.append(ord(item[2]))

    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = instance

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([[int(instance[0]),int(instance[1])]])[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != ord(instance[2]):
        error += 1
    total += 1

#print the error rate
#--> add your Python code here
print("Error rate: ", error/total)





