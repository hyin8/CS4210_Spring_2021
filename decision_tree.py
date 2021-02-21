#-------------------------------------------------------------------------
# AUTHOR: Haowen Yin
# FILENAME: decision_tree.py
# SPECIFICATION: Implement decision tree algorithm ID3
# FOR: CS 4200- Assignment #1
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays
#importing some Python libraries

from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transfor the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]

# Young = 1, Prepresbyopic = 2, Presbyopic = 3,
# Myope = 1, Hypermetrope = 2
# Yes = 1 , No =2
# Normal = 1, Reduced = 2

#--> add your Python code here
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

for instance in db:
  temp = []
  for value in range(4):
    temp.append(attribute_dict[value][instance[value]])
  X.append(temp)
  temp.clear

#transfor the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> addd your Python code here
# Y =
for instance in db:
  if instance[4] == 'Yes':
    Y.append(1)
  elif instance[4] == 'No':
    Y.append(2)

#fiiting the decision tree to the data

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree

tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()

