# Self-Driving Car Engineer Nanodegree
# Deep Learning

# ---- 作者的话开始 ----

# 效果

本程序实现一个可以识别交通标志的神经网络，输入为一张32\*32的彩色图像，输出为43种交通标志的预测结果。

![](https://raw.githubusercontent.com/ypwhs/resources/master/WechatIMG2014.jpeg)

### 数据集地址下载地址：[traffic-sign-data.zip](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580d53ce_traffic-sign-data/traffic-sign-data.zip) [百度云](https://pan.baidu.com/s/1i56YlDF)

数据集预览

![](https://raw.githubusercontent.com/ypwhs/resources/master/WechatIMG2005.jpeg)

# 思路

照着 Keras 的 [cifar10_cnn](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py) 搭就行，训练结果 97% 左右。

我的代码：[Traffic_Signs_Recognition.ipynb](Traffic_Signs_Recognition.ipynb)

参考文献：[Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

# ---- 作者的话结束 ----

## Project: Build a Traffic Sign Recognition Program

**This is a Work In Progress**

### Install

This project requires **Python 3.5** and the following Python libraries installed:

- [Juypyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)

In addition to the above, for those optionally seeking to use image processing software, you may need one of the following:
- [PyGame](http://pygame.org/)
   - Helpful links for installing PyGame:
   - [Getting Started](https://www.pygame.org/wiki/GettingStarted)
   - [PyGame Information](http://www.pygame.org/wiki/info)
   - [Google Group](https://groups.google.com/forum/#!forum/pygame-mirror-on-google-groups)
   - [PyGame subreddit](https://www.reddit.com/r/pygame/)
- [OpenCV](http://opencv.org/)

For those optionally seeking to deploy an Android application:
- Android SDK & NDK (see this [README](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/README.md))

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 3.5 installer and not the Python 2.x installer. `pygame` and `OpenCV` can then be installed using one of the following commands:

Run this command at the terminal prompt to install OpenCV:

**opencv**  
`conda install -c https://conda.anaconda.org/menpo opencv3`

Run this command at the terminal prompt to install PyGame:

**PyGame:**  
Mac:  `conda install -c https://conda.anaconda.org/quasiben pygame`
Windows: `conda install -c https://conda.anaconda.org/tlatorre pygame`
Linux:  `conda install -c https://conda.anaconda.org/prkrekel pygame`

### Code

A template notebook is provided as `Traffic_Signs_Recognition.ipynb`. While no code is included in the notebook, you will be required to use the notebook to implement the basic functionality of your project and answer questions about your implementation and results. 

### Run

In a terminal or command window, navigate to the project directory that contains this README and run the following command:

```bash
jupyter notebook Traffic_Signs_Recognition.ipynb
```

This will open the Jupyter Notebook software and notebook file in your browser.


### Data

1. Download the dataset (2 options)
    - You can download the pickled dataset in which we've already resized the images to 32x32 [here](https://drive.google.com/drive/folders/0B76KYRlYCyRzYjItVFU4aV91b2c).
    - (Optional). You could also download the dataset in its original format by following the instructions [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). We've included the notebook we used to preprocess the data [here](./Process-Traffic-Signs.ipynb).


