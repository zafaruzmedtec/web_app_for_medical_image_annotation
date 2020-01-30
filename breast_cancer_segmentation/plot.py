import seaborn as sea
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = './train_log_43iter_500epoch_2batchsize_1e4lr_binarycross_6conv_256.xlsx'
dataframe = pd.read_excel(io=path)

plt.close('all')
plt.figure()
sea.lineplot(data=dataframe, dashes=True)

mov_avg = np.convolve(a=dataframe['dice_coef'], v=np.ones(20)/20, mode='same')
plt.plot(np.arange(10, len(mov_avg)-10), mov_avg[10:-10])
plt.grid()
plt.show()

