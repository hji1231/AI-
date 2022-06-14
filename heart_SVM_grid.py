import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('../archive/heart.csv') 
data = pd.get_dummies(data, columns = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope'])

X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

svm = SVC()

param = {"C": [0.001,0.01,0.1,1,10,100,1000],
         "gamma": [1000,100,10,1,0.1,0.01,0.001,0.0001]}

grid = GridSearchCV(svm, param_grid=param, scoring='accuracy', refit=True, verbose=1)
grid.fit(X_train, y_train)

print("최적의 파라미터 : ", grid.best_params_)

best = grid.best_estimator_
pred = best.predict(X_test)
print("테스트 집합에 대한 정확률 : ",accuracy_score(y_test,pred))