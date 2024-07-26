from numpy import loadtxt
from keras.models import model_from_json

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)  #open('model.json').read()
model.load_weights('model.h5')

predictions = model.predict(X)
# print(predictions)

for i in range(10,15):
    print('%s => %d (expected %d)' % (X[i], predictions[i], y[i]))