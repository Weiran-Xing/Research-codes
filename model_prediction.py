import numpy as np
from osgeo import gdal
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation, Reshape
from sklearn.ensemble import RandomForestClassifier

# Read the land use data
file_land00 = '00.tif'
data_land00 = gdal.Open(file_land00)

file_land01 = '01.tif'
data_land01 = gdal.Open(file_land01)

file_land02 = '02.tif'
data_land02 = gdal.Open(file_land02)

file_land03 = '03.tif'
data_land03 = gdal.Open(file_land03)

file_land04 = '04.tif'
data_land04 = gdal.Open(file_land04)

file_land05 = '05.tif'
data_land05 = gdal.Open(file_land05)

file_land06 = '06.tif'
data_land06 = gdal.Open(file_land06)

file_land07 = '07.tif'
data_land07 = gdal.Open(file_land07)

file_land08 = '08.tif'
data_land08 = gdal.Open(file_land08)

file_land09 = '09.tif'
data_land09 = gdal.Open(file_land09)

file_land10 = '10.tif'
data_land10 = gdal.Open(file_land10)

file_land11 = '11.tif'
data_land11 = gdal.Open(file_land11)

file_land12 = '12.tif'
data_land12 = gdal.Open(file_land12)

file_land13 = '13.tif'
data_land13 = gdal.Open(file_land13)

file_land14 = '14.tif'
data_land14 = gdal.Open(file_land14)

im_height = data_land00.RasterYSize
im_width = data_land00.RasterXSize
# Read the data as array
im_data_land00 = data_land00.ReadAsArray(0, 0, im_width, im_height) 
im_data_land01 = data_land01.ReadAsArray(0, 0, im_width, im_height)  
im_data_land02 = data_land02.ReadAsArray(0, 0, im_width, im_height)
im_data_land03 = data_land03.ReadAsArray(0, 0, im_width, im_height)  
im_data_land04 = data_land04.ReadAsArray(0, 0, im_width, im_height) 
im_data_land05 = data_land05.ReadAsArray(0, 0, im_width, im_height)  
im_data_land06 = data_land06.ReadAsArray(0, 0, im_width, im_height)
im_data_land07 = data_land07.ReadAsArray(0, 0, im_width, im_height)  
im_data_land08 = data_land08.ReadAsArray(0, 0, im_width, im_height)
im_data_land09 = data_land09.ReadAsArray(0, 0, im_width, im_height)  
im_data_land10 = data_land10.ReadAsArray(0, 0, im_width, im_height)
im_data_land11 = data_land11.ReadAsArray(0, 0, im_width, im_height)  
im_data_land12 = data_land12.ReadAsArray(0, 0, im_width, im_height)  
im_data_land13 = data_land13.ReadAsArray(0, 0, im_width, im_height)  
im_data_land14 = data_land14.ReadAsArray(0, 0, im_width, im_height)  

number = 0
# Calculate the pixel number of Dongduan
for row in range(12,im_height-12):
    for col in range(12,im_width-12):
            if im_data_land00[row][col] != 0 :
                number = number + 1

print("number of DongGuan:\n",number)

# Get the processed data
# The data named"ample_final_XX_2L.txt" is the features which contains 
# initial land use state, neighborhood proportion, slope, elevation and distance-based variables.
# Initial land use state, neighborhood proportion are calculated by each year's data.
# Slope, elevation and distance-based variables of each year are same.

# Data of "Labels_XX.txt" is the label which represents land use state of next time step, using one-hot enconding.
Sample_final_00_2L = np.loadtxt('Sample_final_00_2L.txt')
Label_Samples_00 = np.loadtxt('Labels_00.txt')

Sample_final_01_2L = np.loadtxt('Sample_final_01_2L.txt')
Label_Samples_01 = np.loadtxt('Labels_01.txt')

Sample_final_02_2L = np.loadtxt('Sample_final_02_2L.txt')
Label_Samples_02 = np.loadtxt('Labels_02.txt')

Sample_final_03_2L = np.loadtxt('Sample_final_03_2L.txt')
Label_Samples_03 = np.loadtxt('Labels_03.txt')

Sample_final_04_2L = np.loadtxt('Sample_final_04_2L.txt')
Label_Samples_04 = np.loadtxt('Labels_04.txt')

Sample_final_05_2L = np.loadtxt('Sample_final_05_2L.txt')
Label_Samples_05 = np.loadtxt('Labels_05.txt')

Sample_final_06_2L = np.loadtxt('Sample_final_06_2L.txt')
Label_Samples_06 = np.loadtxt('Labels_06.txt')

Sample_final_07_2L = np.loadtxt('Sample_final_07_2L.txt')
Label_Samples_07 = np.loadtxt('Labels_07.txt')

Sample_final_08_2L = np.loadtxt('Sample_final_08_2L.txt')
Label_Samples_08 = np.loadtxt('Labels_08.txt')

Sample_final_09_2L = np.loadtxt('Sample_final_09_2L.txt')
Label_Samples_09 = np.loadtxt('Labels_09.txt')

Sample_final_10_2L = np.loadtxt('Sample_final_10_2L.txt')
Label_Samples_10 = np.loadtxt('Labels_10.txt')

LSTM_data10years_unpro = np.concatenate((Sample_final_00_2L, Sample_final_01_2L, Sample_final_02_2L, Sample_final_03_2L, Sample_final_04_2L, Sample_final_05_2L, Sample_final_06_2L, Sample_final_07_2L, Sample_final_08_2L, Sample_final_09_2L),axis = 0)
LSTM_data_label10years_unpro = np.concatenate((Label_Samples_00, Label_Samples_01, Label_Samples_02, Label_Samples_03, Label_Samples_04, Label_Samples_05, Label_Samples_06, Label_Samples_07, Label_Samples_08, Label_Samples_09),axis = 0)

# Prepare the input data for LSTM
def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []

    for i in range(length):
        random_list.append(random.randint(start, stop))
    return np.array(random_list)  # This function is used to get a random array


random_list = random_int_list(0,LSTM_data10years_unpro.shape[0]-1,int(LSTM_data10years_unpro.shape[0]* 0.2))
LSTM_data10years = np.zeros((int(LSTM_data10years_unpro.shape[0]* 0.2),25))
LSTM_data_label10years = np.zeros((int(LSTM_data10years_unpro.shape[0]* 0.2),1))
for i in range(int(LSTM_data10years_unpro.shape[0]* 0.2)):
    temp = random_list[i]
    LSTM_data10years[i]= LSTM_data10years_unpro[temp]
    LSTM_data_label10years[i] = LSTM_data_label10years_unpro[temp]


# Division for training sets and test sets
train_num10years = int(LSTM_data10years.shape[0] * 0.7)
test_num10years = LSTM_data10years.shape[0] - train_num10years

LSTM_data_train10years = np.zeros((train_num10years, 25))
LSTM_data_test10years = np.zeros((test_num10years, 25))
LSTM_data_train_label10years = np.zeros((train_num10years, 1))
LSTM_data_test_label10years = np.zeros((test_num10years, 1))
for i in range(train_num10years):
    LSTM_data_train10years[i] = LSTM_data10years[i]
    LSTM_data_train_label10years[i] = LSTM_data_label10years[i]
for j in range(train_num10years, LSTM_data10years.shape[0]):
    LSTM_data_test10years[j - train_num10years] = LSTM_data10years[j]
    LSTM_data_test_label10years[j - train_num10years] = LSTM_data_label10years[j]

# One-hot encoding
LSTM_data_train_label10years = np_utils.to_categorical(LSTM_data_train_label10years, num_classes=5)
LSTM_data_test_label10years = np_utils.to_categorical(LSTM_data_test_label10years, num_classes=5)

# Convert data format for LSTM
LSTM_data_train10years = np.reshape(LSTM_data_train10years, (LSTM_data_train10years.shape[0], LSTM_data_train10years.shape[1], 1))
LSTM_data_test10years = np.reshape(LSTM_data_test10years, (LSTM_data_test10years.shape[0], LSTM_data_test10years.shape[1], 1))


#Build the LSTM model
model = Sequential()
layers = [25, 30, 15, 5]

model.add(LSTM(
    layers[1],
    input_shape=(None,1),
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    layers[2],
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    layers[3]))
model.add(Activation('softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
model.fit(LSTM_data_train10years, LSTM_data_train_label10years, batch_size=64, epochs=8, validation_split=0.1)

print('\nTesting ------------')
loss10years, accuracy10years = model.evaluate(LSTM_data_test10years, LSTM_data_test_label10years)
print('test loss10years: ', loss10years)
print('test accuracy10years: ', accuracy10years)

# Prepare the data for RF
Sample_final_08_RF = np.loadtxt('Sample_final_08_2L.txt')

# Label_final_08_RF
Label_change_08_RF = np.zeros((number,1))
index_change_08 = 0
for row in range(12,im_height-12):
    for col in range(12,im_width-12):
        if im_data_land08[row][col] != 0:
            if im_data_land08[row,col] != im_data_land09[row,col]:
                # The label of nochange is 0£¬the label of change is 1
                Label_change_08_RF[index_change_08,0] = 1
                
            index_change_08 = index_change_08 + 1
            
# Build and train the RF model
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(Sample_final_08_RF, Label_change_08_RF)

# Predict the land use will change or not in the future
Sample_final_10_RF = np.loadtxt('Sample_final_10_2L.txt')
Label_10_RF = rfc.predict(Sample_final_10_RF)

index = 0
change_num_10 = 0
for row in range(12,im_height-12):
    for col in range(12,im_width-12):
        if im_data_land10[row][col] != 0:    
            if Label_10_RF[index] == 1:
                change_num_10 += 1
                
            index = index + 1           
print("change_num_10:",change_num_10)

# Make the prediction of LSTM
Sample_DL_10_final = Sample_final_10_2L

# Convert data format for LSTM
input_10 = np.reshape(Sample_DL_10_final, (Sample_DL_10_final.shape[0], Sample_DL_10_final.shape[1], 1))
predict = model.predict(input_10)

predict_out = np.zeros((predict.shape[0], predict.shape[1]))
# Set the theshold
theshold = 0.8
Change_sum = 0
for i in range(predict.shape[0]):
    for j in range(predict.shape[1]):
        if predict[i][j] >= theshold:
            predict_out[i][j] = 1
            Change_sum = Change_sum + 1
        else:
            predict_out[i][j] = 0

print("The number of predict > theshold:", Change_sum)

# Get the prediction label
Label_predict = np.zeros((number, 1))

lab_num_predict = 0
index = 0
for row in range(12,im_height-12):
    for col in range(12,im_width-12):
        if im_data_land10[row][col] != 0:
            Label_predict[lab_num_predict][0] = im_data_land10[row][col]
                
            if Label_10_RF[lab_num_predict] == 1:              
                for j in range(predict.shape[1]):
                    if predict_out[index][j] == 1:
                        Label_predict[lab_num_predict][0] = j
                index += 1
            lab_num_predict += 1 
            
            
# Get the prediction results
data_new = np.zeros((im_height, im_width))
index = 0
for row in range(12,im_height-12):
    for col in range(12,im_width-12):
        if im_data_land10[row][col] != 0:
            data_new[row][col] = Label_predict[index][0]
            index = index + 1

same_label_origin = 0
for row in range(12,im_height-12):
    for col in range(12,im_width-12):
        if im_data_land10[row][col] != 0:
            if im_data_land10[row][col] == im_data_land11[row][col]:
                same_label_origin = same_label_origin + 1

print("The same label between im_data_land10 and im_data_land11 = ", same_label_origin)

same_label = 0
for row in range(12,im_height-12):
    for col in range(12,im_width-12):
        if im_data_land10[row][col] != 0:
            if im_data_land10[row][col] == data_new[row][col]:
                same_label = same_label + 1

print("The same label between im_data_land10 and data_new = ", same_label)

same = 0
for row in range(12,im_height-12):
    for col in range(12,im_width-12):
        if im_data_land10[row][col] != 0:
            if im_data_land11[row][col] == data_new[row][col]:
                same = same + 1

print("The same label between im_data_land11 and data_new = ", same)
print("the accuracy of predict is:",same/number)

data_new_outtxt = 'Prediction_results.txt'
np.savetxt(data_new_outtxt, data_new, fmt='%s', newline='\n')
