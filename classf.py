from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_folder = r"C:\skin_disease_dataset\train"
test_folder = r"C:\skin_disease_dataset\test"

x_train = train_datagen.flow_from_directory(
    train_folder,
    target_size=(64, 64),
    batch_size=16
)

x_test = test_datagen.flow_from_directory(
    test_folder,
    target_size=(64, 64),
    batch_size=16
)

num_classes = x_train.num_classes

model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # input layer of CNN
model.add(Dense(units=128, kernel_initializer="uniform", activation="relu"))
model.add(Dense(units=100, kernel_initializer="uniform", activation="relu"))
model.add(Dense(units=num_classes, kernel_initializer="uniform", activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(
    x_train,
    steps_per_epoch=48,
    validation_data=x_test,
    validation_steps=8,
    epochs=200
)

model.save("skin_proj.h5")
