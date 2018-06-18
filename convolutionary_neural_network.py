from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


#Initialize the Convolutionary Neural Networks
classifier = Sequential()

# Step 1 - Convolution
# 32 is the number of filters or feature detectors
# the feature detector is 3 pixels by 3 pixels
# 64 by 64 is the number of pixels you want to process, you can put 128 128 or even 2048 by 2048 but that is going to demand too much computing resources
classifier.add(Convolution2D(32,3,3, input_shape=(64, 64, 3), activation='relu'))

# step 2 - Pooling
# 2 by 2 indicates the size of each pools of the pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# add a second convolutionary network to increase the accuracy in the test set
# there is no need to have an input_shape because we receive the data from the MaxPooling
classifier.add(Convolution2D(32,3,3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# step 3 - Flattening
classifier.add(Flatten())

# step 4 - Full connection
# the ouput_dim in this case was the double of the input_shape
classifier.add(Dense(output_dim = 128, activation='relu'))

# ouptput_dim = 1 means the result is either a cat or a dog
classifier.add(Dense(output_dim = 1, activation='sigmoid'))

# compiling the CNN
# adam algorithm contains the stochastic gradient descent
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
# image augmentation allows us to enrich our training set without adding more images

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#for better accuracy, you can increase the target size which means to increase the number of pixels in the images
training_set = train_datagen.flow_from_directory(
        'C:\\Users\\abautista\\Desktop\\Machine_Learning_AZ_Template_Folder\\Part 8 - Deep Learning\\Section 40 - Convolutional Neural Networks (CNN)\\Convolutional-Neural-Networks\\dataset\\training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'C:\\Users\\abautista\\Desktop\\Machine_Learning_AZ_Template_Folder\\Part 8 - Deep Learning\\Section 40 - Convolutional Neural Networks (CNN)\\Convolutional-Neural-Networks\\dataset\\test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000, #8000 is the number of data in our training set
        epochs=25,
        validation_data=test_set,
        validation_steps=2000) # 2000 is the number of data in our test set
