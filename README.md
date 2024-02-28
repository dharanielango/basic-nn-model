# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:Dharani.E
### Register Number:212221230021
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('EX1').sheet1

data = worksheet.get_all_values()

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'input':'float'})
dataset1 = dataset1.astype({'output':'float'})
dataset1.head(20)

import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X = dataset1[['input']].values
y = dataset1[['output']].values
X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train1 = scaler.transform(X_train)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(units=2,activation='relu',input_shape=[1]),
    Dense(units=2,activation='relu'),
    Dense(units=1)
])

model.compile(optimizer='rmsprop',loss='mse')
model.fit(X_train1,y_train,epochs=2000)

loss= pd.DataFrame(model.history.history)
loss.plot()

X_test1 =scaler.transform(X_test)
model.evaluate(X_test1,y_test)

X_n1=[[10]]
X_n1_1=scaler.transform(X_n1)
model.predict(X_n1_1)





```
## Dataset Information

![d4](https://github.com/dharanielango/basic-nn-model/assets/94530523/611bfdea-397c-4fdd-bc0f-746e4bf3cecf)


## OUTPUT

### Training Loss Vs Iteration Plot

![d2](https://github.com/dharanielango/basic-nn-model/assets/94530523/c473391b-6a06-43b1-999c-ac755dc71789)

### Epoch Training 

![d3](https://github.com/dharanielango/basic-nn-model/assets/94530523/317e4f1c-d474-4a05-8e48-2f0dcb8cef59)


### Test Data Root Mean Squared Error


![d1](https://github.com/dharanielango/basic-nn-model/assets/94530523/1274f623-c3a1-4864-8a04-cf0f4a3cdaaa)

### New Sample Data Prediction

![d](https://github.com/dharanielango/basic-nn-model/assets/94530523/7e9f4d3e-2683-461b-99f9-6a056ef628ac)


## RESULT

Thus the neural network regression model for the given dataset is executed successfully.
