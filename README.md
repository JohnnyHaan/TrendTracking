# Time Series Data Analyze

IN this project we use some Chinese stock market information to precdict or guess the upward or downward. We use a cnn neural network to train and test. and surely with data cleaning part which use SOME SIMPILFY method include deleting or filing some NAN data, rescale some bigger data and so on.

BUT the MOST IMPORTANT things which I must remind or warn you, this project only use to study Time Series Data Analyze, and DO NOT BELIVE the trend which the model predicted

## Contents

./README
./anaconda_packages_list.txt
./predict/
./predict/data_sets/   ### some demo training ,testing and predict data
./predict/utils_data/   ### data processing ATTENTION when you use the numpy np.std(X, axis=0, ddof=1), please make sure the ddof method, in this project we use ddof default value(as ZER0)
./predict/utils_models/  ### convolution neural network designing part
./predict/checkpoints-cnn/  ### the path of the saving model
./predict/checkpoints-cnn-max/  ### the highest accuray of the saving model when validating during the training

# Enviroment

We use a notebook to train our Time Series Data Analyze Project and The Basic Information As bellowing

Intel i5-3317U CPU @ 1.70GHZ
RAM 4G
x64 WIN8

ANACONDA 1.19.12
Please Reference ./anaconda_packages_list.txt