# Ex.No: 13 Learning - Use Supervised Learning
### DATE:                                                                            
### REGISTER NUMBER: 212222040128
### AIM: 
To write the program to train the classifier for Diabetes.
###  Algorithm:
Step 1: Import packages <br>
Step 2: Get the data<br>
Step 3: Split the data <br>
Step 4: Scale the data <br>
Step 5: Instantiate model <br>
Step 6: Create a function for gradio <br>
Step 7: Print Result <br>
### Program:
```
import numpy as np
import pandas as pd
pip install gradio
pip install typing-extensions --upgrade
pip install --upgrade typing
pip install typing-extensions --upgrade
import gradio as gr
data = pd.read_csv('/content/diabetes.csv')
data.head()
print(data.columns)
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']
print(x[:5])
#split data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x,y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(max_iter=1000, alpha=1)
model.fit(x_train, y_train)
print("Model Accuracy on training set:", model.score(x_train, y_train))
print("Model Accuracy on Test Set:", model.score(x_test, y_test))

def diabetes(Pregnancies, Glucose, Blood_Pressure, SkinThickness, Insulin, BMI,Diabetes_Pedigree, Age):
    x = np.array([Pregnancies,Glucose,Blood_Pressure,SkinThickness,Insulin,BMI,Diabetes_Pedigree,Age])
    prediction = model.predict(x.reshape(1, -1))
    if(prediction==0):
      return "NO"
    else:
      return "YES"

outputs = gr.Textbox()
app = gr.Interface(fn=diabetes, inputs=['number','number','number','number','number','number','number','number'], outputs=outputs,description="Detection of Diabeties")
app.launch(share=True)

```

### Output/Plan:

#### 1. Dataset
![image](https://github.com/AaronDominic/AI_Lab_2023-24/assets/143015231/ac8a0336-1652-4d87-b843-b7cbbc29c8b2)

#### 2. Accuracy
![image](https://github.com/AaronDominic/AI_Lab_2023-24/assets/143015231/2353b5a1-b447-43d3-a769-11db565b73f6)

#### 3. Output Result
![image](https://github.com/AaronDominic/AI_Lab_2023-24/assets/143015231/15d88535-80f2-4b43-9c21-002ac0462bfb)



### Result:
Thus the system was trained successfully and the prediction was carried out.
