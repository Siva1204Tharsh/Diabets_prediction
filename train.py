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

from numpy import loadtxt #row data  Load panna 
from keras.models import Sequential #   Sequential model (input layer -> hidden layer -> output layer)
from keras.layers import Dense # Fully Connected Layer
from keras.models import model_from_json #model into json file save and load it ///-pickle - binary file
 
dataset=loadtxt('pima-indians-diabetes.csv',delimiter=',') # csv file - comma separater value # load the dataet  
# print(dataset) 
X=dataset[:,0:8] #input layer [raw data,column 0-7]
y=dataset[:,8] #output layer [target,column 8]
# print("X input:",X)
# print("y output:",y)
model=Sequential() # Sequential model load 
model.add(Dense(12,input_dim=8,activation='relu')) #Dense - fully connected layer
model.add(Dense(8,activation='relu'))  #relu -> sigmoid
model.add(Dense(1,activation='sigmoid')) #sigmoid is binary classification

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) #binary_crossentropy

##Model training
model.fit(X,y,epochs=30,batch_size=6)# accuracy increases after 30 epochs or over fiting the data

##Evaluation
_, accuracy = model.evaluate(X,y) # output evaluation 2 => values accuracy and loss
print('Accuracy: %.2f' % (accuracy*100))

##model Save
model_json=model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.weights.h5') # 
print("Saved model to disk")