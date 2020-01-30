from model import *
from data import *

# test your model and save predicted results
test_img_path = './data/inbreast/test_final/image/'
groundtruth_path = './data/inbreast/test_final/label/'
predicted_path = './data/inbreast/test_final/predicted/'
img_ext = '*.png'

img_names = glob.glob(os.path.join(test_img_path,img_ext))
num_img = len(img_names)
testGene = testGenerator(img_names)

model = unet()
model.load_weights("unet_inbreast_43iter_500epoch_2batchsize_1e4lr_binarycross_6conv_256.hdf5")
results = model.predict_generator(testGene,num_img,verbose=1)
predicted_img = saveResult(predicted_path,results,img_names)


def dice_coef(y_true, y_pred):
    return np.sum(y_pred[y_true==255])*2.0 / (np.sum(y_pred) + np.sum(y_true))
    
groundtruth_name_path_arr = glob.glob(os.path.join(groundtruth_path,img_ext))
predicted_name_path_arr = glob.glob(os.path.join(predicted_path,img_ext))

dice = []
for i in range(len(groundtruth_name_path_arr)):
    groundtruth = io.imread(groundtruth_name_path_arr[i])
    groundtruth = cv2.resize(groundtruth,(int(2560),int(3328)))
    predicted = io.imread(predicted_name_path_arr[i])
    predicted = cv2.resize(predicted,(int(2560),int(3328)))
    dice.append(dice_coef(groundtruth, predicted))
print('Average dice index: ' + str(sum(dice)/len(groundtruth_name_path_arr)))
np.savetxt('test_dice_final.csv', dice, delimiter=',', header='dice_index')
