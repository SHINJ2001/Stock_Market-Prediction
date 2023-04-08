from data_visualization import visualize
from ANFIS.data_model import anfis
#from SVM.data_model import svm
from LSTM.data_model import lstm

df = visualize()
print("Running ANFIS on the dataset -->")
acc1 = anfis(df)
print("Running LSTM on the dataset -->")
acc2 = lstm(df)
print("Running SVM on the dataset -->")
acc3 = svm(df)

print("The MSE values were obtained as follows -->")

print("ANFIS -- " + str(acc1))
print("LSTM -- " + str(acc2))
print("SVM -- " + str(acc3))
