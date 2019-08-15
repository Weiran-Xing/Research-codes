import numpy as np
from osgeo import gdal
from keras.models import Sequential
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D

# Read the land use data of the current year and the next year
file_land00 = '00.tif'
data_land00 = gdal.Open(file_land00)

file_land01 = '01.tif'
data_land01 = gdal.Open(file_land01)


im_height = data_land00.RasterYSize
im_width = data_land00.RasterXSize

# Read the data as array
im_data_land00 = data_land00.ReadAsArray(0, 0, im_width, im_height)
im_data_land01 = data_land01.ReadAsArray(0, 0, im_width, im_height)

# Calculate the pixel number of Dongduan
number = 0
for row in range(12,im_height-12):
    for col in range(12,im_width-12):
            if im_data_land00[row][col] != 0 :
                number = number + 1

print("number of DongGuan:\n",number)

# Divide the land use data with a grid of 25*25
data_sample_00 = np.zeros((number, 25, 25))
data_sample_label_00 = np.zeros((number,1))

number_index = 0
for row in range(12,im_height-12):
    for col in range(12,im_width-12):
        if im_data_land00[row][col] != 0:
            data_sample_label_00[number_index,0] = im_data_land01[row,col]
            for grid_row in range(-12,13):
                for grid_col in range(-12,13):
                    data_sample_00[number_index,12+grid_row,12+grid_col] = im_data_land00[row+grid_row,col+grid_col]
            number_index = number_index + 1

# Prepare the input data for CNN
def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []

    for i in range(length):
        random_list.append(random.randint(start, stop))
    return np.array(random_list)  # This function is used to get a random array

num_of_year = int(number * 0.2)
random_list = random_int_list(0, number - 1, num_of_year)

cnn_data00 = np.zeros((num_of_year, 25, 25))
cnn_data_label00 = np.zeros((num_of_year, 1))

for i in range(0, num_of_year):
    temp = random_list[i]
    cnn_data00[i] = data_sample_00[temp]
    cnn_data_label00[i] = data_sample_label_00[temp]


# Division for training sets and test sets
train_num = int(cnn_data00.shape[0] * 0.7)
test_num = cnn_data00.shape[0] - train_num

cnn_data_train = np.zeros((train_num, 25, 25))
cnn_data_test = np.zeros((test_num, 25, 25))
cnn_data_train_label = np.zeros((train_num, 1))
cnn_data_test_label = np.zeros((test_num, 1))
for i in range(train_num):
    cnn_data_train[i] = cnn_data00[i]
    cnn_data_train_label[i] = cnn_data_label00[i]
for j in range(train_num, cnn_data00.shape[0]):
    cnn_data_test[j - train_num] = cnn_data00[j]
    cnn_data_test_label[j - train_num] = cnn_data_label00[j]
	
# One-hot encoding
cnn_data_train_label = np_utils.to_categorical(cnn_data_train_label, num_classes=5)
cnn_data_test_label = np_utils.to_categorical(cnn_data_test_label, num_classes=5)

# Convert data format for CNN
cnn_data_train= cnn_data_train.reshape(train_num,25,25,1)
cnn_data_test= cnn_data_test.reshape(test_num,25,25,1)

#Build the CNN model
model=Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(25, 25, 1), padding="same") )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),  padding="same"))

model.add(Conv2D(64, (5, 5), padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),  padding="same"))

model.add(Flatten())

model.add(Activation('relu'))
model.add(Dense(5))
model.add(Activation('softmax'))
# Define the optimize
rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])

print('\nTraining-----------')
model.fit(cnn_data_train,cnn_data_train_label,epochs=10,batch_size=32)
model.summary()

print('\nTesting------------')
loss,accuracy=model.evaluate(cnn_data_test,cnn_data_test_label)
print('test loss: ', loss)
print('test accuracy: ', accuracy)

# Extraction of latent spatial features
data_sample_00 = data_sample_00.reshape(data_sample_00.shape[0],25,25,1)
predict_00 = model.predict(data_sample_00)
data_new_outtxt = 'CNN_features_00'
np.savetxt(data_new_outtxt, predict_00, fmt='%s', newline='\n')