# Time Series Data Analyze

IN this project we use some Chinese stock market information to precdict or guess the upward or downward. We use a cnn neural network to train and test. and surely with data cleaning part which use SOME SIMPILFY method include deleting or filing some NAN data, rescale some bigger data and so on.

BUT the MOST IMPORTANT things which I must remind or warn you, this project only use to study Time Series Data Analyze, and DO NOT BELIVE the trend which the model predicted

# Contents

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

# Operation
STEP ONE  
(base) PS D:\AI\Project\stock\GitHub\TrendTracking\predict> conda activate stock  
(stock) PS D:\AI\Project\stock\GitHub\TrendTracking\predict>  
  
STEP TWO  
A few stocks have been download in the data_sets folder, if you just run the demo,please ignore the step two. if you download more stocks just do as bellowing  
(stock) PS D:\AI\Project\stock\GitHub\TrendTracking\predict> python .\get_stock_data.py  
login success!  
2020-04-20  
           code tradeStatus   code_name  
0     sh.000001           1      上证综合指数  
...         ...         ...         ...  
4332  sz.399994           1  中证信息安全主题指数  
[4337 rows x 3 columns]  
processing sh.000001 上证综合指数  
  
STEP THREE  
(stock) PS D:\AI\Project\stock\GitHub\TrendTracking\predict> jupyter notebook  
[I 20:55:01.266 NotebookApp] Serving notebooks from local directory: D:\AI\Project\stock\GitHub\TrendTracking\predict  
[I 20:55:01.266 NotebookApp] The Jupyter Notebook is running at:  
[I 20:55:01.267 NotebookApp] http://localhost:8888/  
[I 20:55:01.271 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).  
  
STEP FOUR  
open the stock_cnn.ipynb with the jupyer notebook  
<img src="image/jupyter.png" alt="drawing" width="50%"/>  
press the fast-forward button  
<img src="image/run.png" alt="drawing" width="50%"/>  