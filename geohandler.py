 
import numpy as np
from sklearn.preprocessing import LabelBinarizer

def data_load(dataset,class_n=3):
	path='/home/lvc/Jorg/igarss/wildfire_fcn/src/patch_extract2/compact/'+dataset+'/'
	x_train=np.load(path+"train_im.npy")
	y_train=np.load(path+"train_label.npy")
	window_len=y_train.shape[1]
	y_train=np.squeeze(y_train[:,int(window_len/2),int(window_len/2)])

	x_test=np.load(path+"val_im.npy")
	y_test=np.load(path+"val_label.npy")
	y_test=np.squeeze(y_test[:,int(window_len/2),int(window_len/2)])


	y_train=array_to_one_hot(y_train,class_n)
	y_test=array_to_one_hot(y_test,class_n)
	print("Y train shape",y_train.shape)

	return (x_train,y_train),(x_test,y_test)

def array_to_one_hot(array,class_n=3):
	label_binarizer = LabelBinarizer()
	label_binarizer.fit(range(class_n))
	return label_binarizer.transform(array)
	
data_load("area3")