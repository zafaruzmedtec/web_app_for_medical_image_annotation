from model import *
from data import *

# Train with data generator
data_gen_args = dict()
myGene = trainGenerator(2,'data/inbreast/train','image','label',data_gen_args,save_to_dir = None) #batch size = 2 to 25 'data/inbreast/generator'
model = unet()

class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

log_filename =  'train_log_43iter_300epoch_2batchsize_1e4lr_binarycross_6conv_256.csv' 

history = LossHistory()
csv_log = callbacks.CSVLogger(log_filename, separator=',', append=True)

#checkpoint = callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    

model_checkpoint = ModelCheckpoint('unet_inbreast_43iter_300epoch_2batchsize_1e4lr_binarycross_6conv_256.hdf5', monitor='loss',verbose=1, save_best_only=True)
callbacks_list = [history, csv_log, model_checkpoint]
model.fit_generator(myGene,steps_per_epoch=43,epochs=300,callbacks=callbacks_list)


## Train with npy file
#imgs_train,imgs_mask_train = geneTrainNpy("data/membrane/train/aug/","data/membrane/train/aug/")
#model.fit(imgs_train, imgs_mask_train, batch_size=2, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

## test your model and save predicted results
# testGene = testGenerator("data/inbreast/test")
# model = unet()
# model.load_weights("unet_inbreast.hdf5")
# results = model.predict_generator(testGene,30,verbose=1)
# saveResult("data/inbreast/test",results)