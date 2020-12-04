import time
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from sklearn.model_selection import GridSearchCV, ParameterGrid
from keras.wrappers.scikit_learn import KerasClassifier
import pandas
import plotly
from tensorflow import keras

numpy.random.seed(42)

# Розмір міні-вибірки
batch_size = 32
# Кількість класів зображень
nb_classes = 10
# Кількість епох навчання 
nb_epoch = 25
# Розмір зображення
img_rows, img_cols = 32, 32
# Кількість каналів: RGB
img_channels = 3

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

def create_model(learn_rate=0.01, beta_1=0.95, momentum=0.9):
  # Створення нейромережевої моделі
  model = Sequential()
  # Перший шар згортки
  model.add(Conv2D(32, (3, 3), padding='same',
                          input_shape=(32, 32, 3), activation='relu'))
  # Друний шар згортки
  model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
  # Перший шар субдискретизаії 
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # Перший шар Dropout
  model.add(Dropout(0.25))

  # Третій шар згортки
  model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
  # Четвертий шар згортки
  model.add(Conv2D(64, (3, 3), activation='relu'))
  # Другий шар субдисктеризації
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # Другий шар Dropout
  model.add(Dropout(0.25))
  # Шар перетворення вхідних даних 
  model.add(Flatten())
  # Повнозв’язний шар
  model.add(Dense(512, activation='relu'))
  # Третій шар Dropout
  model.add(Dropout(0.5))
  # Вихідний шар 
  model.add(Dense(nb_classes, activation='softmax'))

  # Параметри оптимізації
  sgd = SGD(lr=learn_rate, decay=1e-6, momentum=momentum, nesterov=True)
  # adam = Adam(learning_rate=learn_rate, beta_1=beta_1, beta_2=0.999, epsilon=1e-07)
  model.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'])
  return model

# Навчання  моделі
model = create_model()
history = model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_split=0.1,
              shuffle=True,
              verbose=2)

# Оцінка якості навчання на тестових даних
scores = model.evaluate(X_test, Y_test)
print("Accuracy on test data: %.2f%%" % (scores[1]*100))

grid_model = KerasClassifier(build_fn=create_model, verbose=0)

batch_size = [20, 25, 30]
epochs = [20, 25, 30]
learn_rate = [0.005, 0.01, 0.015]
momentum = [0.85, 0.9, 0.95]

grid = dict(epochs=epochs, batch_size=batch_size, learn_rate=learn_rate, momentum=momentum)

t1 = time.time()
scores = []
model_tt = KerasClassifier(build_fn=create_model, verbose=0)
for g in ParameterGrid(grid):
    model_tt.set_params(**g)
    model_tt.fit(X_train, Y_train)
    scores.append(dict(params=g, score=model_tt.score(X_test, Y_test)))
    print('model#',len(scores), scores[-1])

t2 = time.time()

print("Training time:",t2-t1, 'sec')

df = pandas.DataFrame([{**row['params'], **row} for row in scores])
df = df.drop('params', axis=1)
df.sort_values('score')

model_tt.model.save('my_model.h5')  

model = keras.models.load_model('my_model.h5')  
print(model)
model.predict(x) 