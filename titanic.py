#Following minor tweaks are done to test data before using it
#test.csv - added dummy 'Survived' column with all zeros just to make dimensions compatible with training data
#test.csv - added fare value as '60' for Passenger 1044

import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

pid = []

def main():
    #Initial processing
    process_dat('train')
    process_dat('test')
    
    #Reading processed data
    fp1 = csv.reader(open('train_processed.csv', 'rb'))
    fp2 = csv.reader(open('test_processed.csv', 'rb'))
    #Skipping header
    header = np.array(fp1.next())
    header_t = np.array(fp2.next())

    dat = []
    dat_t = []

    for row in fp1:
        dat.append(row)

    for row in fp2:
        dat_t.append(row)

    dat = np.array(dat, dtype=np.float64)
    dat_t = np.array(dat_t, dtype=np.float64)

    m, n = dat.shape
    mt, nt = dat_t.shape

    #Setting value for percent of data for cross validation, here 30%
    cv_per = int(.3 * m)

    #Slicing out data for testing using cross validation, first 30% rows
    dat_cv = dat[:cv_per, :]
    y_cv = dat_cv[:, 0]
    x_cv = dat_cv[:, 1:]

    #Slicing out data for training using cross validation, last 70% rows
    dat_tr = dat[cv_per:, :]
    y_tr = dat_tr[:, 0]
    x_tr = dat_tr[:, 1:]

    #Applying Random Forest classifier
    rforest = RandomForestClassifier(n_estimators = 100)
    rforest = rforest.fit(x_tr, y_tr)
    result_cv_rf = rforest.predict(x_cv)

    print "Efficiency using Random Forest = " + str(cal_perf(result_cv_rf, y_cv))

    x = np.delete(dat_t, 1, 1)

    result = rforest.predict(x)

    result = np.hstack((np.array(pid).reshape(mt, 1), np.array(result).reshape(mt, 1)))

    write_result(result)

#Calculating performance via cross validation for tuning model parameters
def cal_perf(result, y_cv):
    result = np.array(result)
    m = result.size
    result = np.hstack((result.reshape(m, 1), y_cv.reshape(m, 1)))

    corr = 0

    for row in result:
        if (row[0] == row[1]):
            corr += 1

    return float(corr)/m

#Writing predictions to a file
def write_result(result):
    fp2 = open('output.csv', 'wb')
    for row in result:
        fp2.write(str(row[0]) + ',' + str(row[1][0]))
        fp2.write('\n')
    fp2.close()

#Scales values by subtracting the mean and dividing by standard deviation
def scale_data(x):
    #Getting mean column-wise
    mu = x.mean(axis=0)
    #Getting standard deviation column-wise
    sigma = x.std(axis=0)
    #Normalizing data
    x = (x - mu)/sigma
    return x

#Preprocesses data to remove unwanted attributed and fill missing values
def process_dat(dat_type):
    #Reading data
    fp1 = csv.reader(open(dat_type + '.csv', 'rb'))
    #Skipping header
    header = np.array(fp1.next())

    x = []

    for row in fp1:
        x.append(row)

    x = np.array(x)

    global pid

    if dat_type == 'test':
        pid = x[:, 0]

    #Dimensions of data
    m, n = x.shape

    age_sum = 0

    #Calculating average age, taking missing values to be zero
    for row in x:
        if row[5]:
            age_sum += float(row[5])

    avg = int(age_sum/m)

    #Replacing missing age with average age
    for row in x:
        if not row[5]:
            row[5] = avg

    #Selecting particular attributes, skipping - Id, Name, Ticket, Cabin and Embarked
    x = np.hstack((float_transpose(x[:,1], m), float_transpose(x[:,2], m), float_transpose(mapgender(x[:,4]), m), float_transpose(x[:,5], m), float_transpose(x[:,6], m) + float_transpose(x[:,7], m), float_transpose(x[:,9], m)))

    #Adjusting header as per new attributes
    header = np.hstack((header[1:3], header[4:6], np.array(["Relatives"]), header[9]))

    #Writing final data
    fp2 = csv.writer(open(dat_type + '_processed.csv', 'wb'))
    fp2.writerow(list(header))

    for row in x:
        fp2.writerow(row)

#Function to convert Gender value to .1 in case of male and .9 in case of female
def mapgender(x):
    y = []
    for row in x:
        if row == 'male':
            y.append(0.1)
        else:
            y.append(0.9)
    return np.array(y, dtype=np.float64)

#Function to convert attribute columns to numpy array, then converting the values to float and then taking transpose
def float_transpose(x, m):
    return np.array(x, dtype=np.float64).reshape(m, 1) 

if __name__ == "__main__":
    main()
