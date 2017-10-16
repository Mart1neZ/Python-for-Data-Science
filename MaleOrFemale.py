from sklearn import tree
from sklearn import svm
from sklearn.linear_model import SGDClassifier

#[height, weight, shoe size]
X = [[181,80,44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
	[166,65,40], [190,90,47], [175,64,39], [177,70,40], [159,55,37],
	[171,75,42], [181,85,43]]

Y = ['male', 'female', 'female', 'female','male', 'male', 'male', 'female', 'male', 'female', 'male']


#tree
clf = tree.DecisionTreeClassifier()

#svm
clf1 = svm.SVC()

#SGDC
clf2 = SGDClassifier()


#tree
clf = clf.fit(X,Y)

#svm
clf1 = clf1.fit(X,Y)

#SGDC
clf2 = clf2.fit(X,Y)


#tree
predicition1 = clf.predict([[190,70,43]])

#svm
predicition2 = clf1.predict([[190,70,43]])

#SGDC
predicition3 = clf2.predict([[190,70,43]])


#print tree
print("From tree: ", predicition1, "\n")

#print svm
print("From svm: ", predicition2, "\n")

#print SGDC
print("From SGDC: ", predicition3, "\n")