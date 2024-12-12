# TAMAMo
Thanks for using Token Aligned Maimai Analyzer Model (TAMAMo). This readme file will guide you to utilize the developed tools about our model. Your working directory should be "some_path/TAMAMO/"

First we will walk through what is there in each file.
## checkpoints
This folder save the parameter of the model.
## configs
This folder store the config file including model parameter, training setting and etc. It will be later imported to training and testing process.
## data
This folder holds the token file that will be used for training and testing process. The ratio of the data is manually adjusted for better model performance.
## model
This folder holds the model file. It will also be imported later for training and testing.
## one_file_demo
This folder is used for demo purpose. I have included all plotted result under one_demo_file/result/. Other files/folder under this folder are either necessary file for training/testing or for holding data generated during training.
## std_tokens_lib
This folder also holds token file. Instead, the ratio of different class of data is not adjusted.
## tools
This folder holds the very majority of the code.
- ChartHanlder.py: all tools needed for decomposing a chart from a piece of string with several lines to tokens.
- ChartRatiingBinder.py: bind the chart file and its corresponding rating.
- ChartStats.py: the PyTorch dataset. Preparing data to feed the model.
- test.py: for measuring the performance of the model
- train.py: training the model
# How to train and test a model
##### To execute train.py, run the code in the terminal and attach arguments.
    python tools\train.py --config ${CONFIG_FILE} --device ${DEVICE_TYPE} --save_dir ${OUTPUT_FILE} --valid ${BOOL} --lossplot ${BOOL} --dataset {TOKEN_FILE}
    
##### To execute test.py, run the code in the terminal and attach arguments.
    python tools\test.py --config ${CONFIG_FILE} --device ${DEVICE_TYPE} --std ${STD_TOKEN_FILE} --checkpoint {CHECKPOINT_FILE} --dist {PLOT_OUTPUT}
