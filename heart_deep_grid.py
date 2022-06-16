import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

data = pd.read_csv('../archive/heart.csv') 
data = pd.get_dummies(data, columns = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope'])

X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

def build_model(n_hidden_Layer, n_hidden_Layer_sizes, learning_rate):
    model=Sequential()
    model.add(Dense(units = 512, input_shape=(20,)))
    for layer in range(n_hidden_Layer):
        model.add(Dense(n_hidden_Layer_sizes, activation='relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr= learning_rate), metrics = ['accuracy'])
    return model

keras_model=keras.wrappers.scikit_learn.KerasClassifier(build_model)

param={
    'n_hidden_Layer' : [0, 1, 2, 3, 4, 5],
    'n_hidden_Layer_sizes': [64, 128, 256, 512],
    'learning_rate' : [0.1, 0.01, 0.001, 0.0001]
    }

grid = GridSearchCV(keras_model, param_grid=param, scoring='accuracy', refit=True, verbose=2)
grid.fit(X_train, y_train, epochs=20, batch_size = 256,
         validation_data=(X_test, y_test))

print("최적의 파라미터 : ", grid.best_params_)

best = grid.best_estimator_.model
res = best.evaluate(X_test, y_test, verbose = 0)
print("정확률 : ", res[1])