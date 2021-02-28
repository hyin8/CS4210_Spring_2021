#-------------------------------------------------------------------------
# AUTHOR: Haowen Yin
# FILENAME: decision_tree.py
# SPECIFICATION: Training and Testing data
# FOR: CS 4200- Assignment #2
# TIME SPENT: 1 hour 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

Age = {
  'Young': 1,
  'Prepresbyopic': 2,
  'Presbyopic': 3,
}
Spectacle = {
  'Myope': 1,
  'Hypermetrope':2
}
Astigmation = {
  'Yes': 1,
  'No': 2
}
Tear={
  'Normal': 1,
  'Reduced': 2
}

attribute_dict = [Age,Spectacle,Astigmation,Tear]

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    for instance in dbTraining:
        temp = []
        for value in range(4):
            temp.append(attribute_dict[value][instance[value]])
        X.append(temp)
        temp.clear

    #transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    for instance in dbTraining:
        if instance[4] == 'Yes':
            Y.append(1)
        elif instance[4] == 'No':
            Y.append(2)

    #loop your training and test tasks 10 times here
    accuracy_ary = []
    for i in range (10):

        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        #--> add your Python code here
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append (row)

        class_predicted = []
        class_true = []
        confusion_matrix = [[0]*2 for i in range(2)]

        for data in dbTest:
            #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            #class_predicted = clf.predict([[3, 1, 2, 1]])[0]           -> [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here

            temp = []
            for value in range(4):
                temp.append(attribute_dict[value][data[value]])
            
            predict_value = clf.predict([temp])[0]
            class_predicted.append(predict_value)

            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            true_label = -1
            if data[4] == "Yes":
                true_label = 1
            elif data[4] == "No":
                true_label = 2

            confusion_matrix[true_label-1][predict_value-1] += 1

        accuracy = (confusion_matrix[0][0]+confusion_matrix[1][1]) /(confusion_matrix[0][0]+confusion_matrix[0][1]+confusion_matrix[1][0]+confusion_matrix[1][1]) 
        accuracy_ary.append(accuracy)

        #find the lowest accuracy of this model during the 10 runs (training and test set)
        #--> add your Python code here
        accuracy_ary.sort()

    #print the lowest accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that:
         #final accuracy when training on contact_lens_training_1.csv: 0.2
         #final accuracy when training on contact_lens_training_2.csv: 0.3
         #final accuracy when training on contact_lens_training_3.csv: 0.4
    #--> add your Python code here
    print("final accuracy when training on ", ds, ": ", accuracy_ary[0])
    accuracy_ary.clear()




