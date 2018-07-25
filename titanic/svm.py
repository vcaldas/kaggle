from sklearn import svm
import pandas as pd
import numpy as np

test = pd.read_csv("./titanic/data/test.csv")
train = pd.read_csv("./titanic/data/train.csv")
drop_columns = ["PassengerId", "Name", "Embarked", "Cabin", "Ticket", "Age", "Fare"]
train_drop_columns = ["Survived"] + drop_columns

# Labels from the training set
train_labels = train['Survived']

# Remove columns that are string or missing too much information + classification label from the feature array
train = train.drop(columns=train_drop_columns)
# Remove the string columns from the test feature array
test = test.drop(columns=drop_columns)

# Replace 'female' and 'male' to 0 and 1
train.loc[train.Sex == 'female', 'Sex'] = 0
train.loc[train.Sex == 'male', 'Sex'] = 1
test.loc[test.Sex == 'female', 'Sex'] = 0
test.loc[test.Sex == 'male', 'Sex'] = 1

clf = svm.SVC()
clf.fit(train.values.tolist(), train_labels.tolist())
classification = clf.predict(test.values.tolist())

result = pd.DataFrame(np.array(classification), columns=["Survived"])
result.index.name = 'PassengerId'
result.index += 892
result.to_csv(path_or_buf="./titanic/results/moya-svm-result077.csv", header=True, index=True)
