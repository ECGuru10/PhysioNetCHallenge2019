import csv
import numpy as np
import glob



def load_challenge_data_with_lbls(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')

    # Ignore SepsisLabel column if present.
#    if column_names[-1] == 'SepsisLabel':
#        column_names = column_names[:-1]
#        data = data[:, :-1]

    return data


with open("../train_test_data2/test_ind.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        test_ind=row
        
with open("../train_test_data2/train_ind.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        train_ind=row
        
        
XTrain=[]
YTrain=[]
XTest=[]
YTest=[]

file=glob.glob("../training_original2/*.psv")


for k in train_ind:
    data=load_challenge_data_with_lbls(file[int(k)-1])
    data=data.astype(np.float32)
    XTrain.append(data[:,:-1])
    YTrain.append(data[:,-1:])
    
for k in test_ind:
    data=load_challenge_data_with_lbls(file[int(k)-1])
    data=data.astype(np.float32)
    XTest.append(data[:,:-1])
    YTest.append(data[:,-1:])
    
np.save('../train_test_data2/XTrain.npy',XTrain)
np.save('../train_test_data2/YTrain.npy',YTrain)
np.save('../train_test_data2/XTest.npy',XTest)
np.save('../train_test_data2/YTest.npy',YTest)
    
    
    
    