'''
1. Number of times pregnant
2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. Diastolic blood pressure (mm Hg)
4. Triceps skin fold thickness (mm)
5. 2-Hour serum insulin (mu U/ml)
6. Body mass index (weight in kg/(height in m)^2)
7. Diabetes pedigree function
8. Age (years)
9. Class variable (0 or 1)

'''

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

dataset=loadtxt('pima-indians-diabetes.csv',delimiter=',')
# print(dataset)
X=dataset[:,0:8] #input layer [raw data,column 0-7]
y=dataset[:,8] #output layer [target,column 8]
# print("X input:",X)
# print("y output:",y)
model=Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))  #relu -> sigmoid
model.add(Dense(1,activation='sigmoid')) #sigmoid is binary classification

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) #binary_crossentropy

##Model training
model.fit(X,y,epochs=30,batch_size=10)# accuracy increases after 30 epochs or over fiting the data

##Evaluation
_, accuracy = model.evaluate(X,y)
print('Accuracy: %.2f' % (accuracy*100))

##model Save
model_json=model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save("model.h5")
print("Saved model to disk")