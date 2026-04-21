# Segmentation-of-polyps

## 1.Data Related Notes：

The training and test sets for this study and the weights associated with the model (the best weights within 300 rounds) can be downloaded at:https://drive.google.com/drive/folders/1scOMQk3PPxkKcLRNLMJrHR_1NpZOsvRV?usp=sharing

You can run the code using the simple data in Sample-dataset.h5

## 2.Notes on the file:

Here are the files and instructions in the code folder:

	│  How to use h5 files.ipynb  （The usage method of h5 files）
	|  Sample-dataset.h5（Sample data）
	│  The random numbers partition Train and Val.ipynb（Split the validation set from the training set）
	│  Train.txt（The split training set）
	│  Val.txt（The split validation set）
	│
	├─Deep learning Mode（Deep Learning Models folder）
	│      FPN+V3.ipynb（Code for the FPN+V3 model）
	│      Linknet+V3.ipynb（Code for the Linknet+V3 model）
	│      PraNet.ipynb（Code for the PraNet model）
	│      PSPnet+V3.ipynb（Code for the PSPnet+V3 model）
	│      SegNet-Simple.ipynb（Code for the Simplify SegNet model code）
	│      SegNet.ipynb（Code for the SegNet model）
	│      U-net++.ipynb（Code for the U-net++ model）
	│      U-net+V3.ipynb（Code for the U-net+V3 model）
	│
	└─Digital Image Methods（Digital image enhancement Techniques methods folder）
	├─1.Edge detection（Edge detection technology）
	│      1. img+edge Enhancing images.ipynb（Edge image + Code for the enhanced image obtained from the original image）
	│      1.BHPF (Extract edges in batches).ipynb（Butterworth high-pass filter obtains the code of the edge image）
	│      1.Canny (Extract edges in batches).ipynb（The Canny operator obtains the code of the edge image）
	│      1.GHPF (Extract edges in batches).ipynb（Gaussian high-pass filter obtains the code of the edge image）
	│      1.IHPF (Extract edges in batches).ipynb（Code for obtaining the edge image for the ideal high-pass filter operator）
	│      1.Marr (Extract edges in batches).ipynb（Marr operator obtains the code of the edge image）
	│      1.Prewitt (Extract edges in batches).ipynb（Prewitt operator obtains the code of the edge image）
	│      1.Roberts (Extract edges in batches).ipynb（Roberts operator obtains the code of the edge image）
	│      1.Sobel (Extract edges in batches).ipynb（Sobel operator gets the code of the edge image）
	│
	├─2.Highlight repair（Highlight repair technique）
	│      2.Highlight repair(Dynamic threshold percentage+FMM).ipynb（Percentage dynamic threshold +FMM technique fixes the code for highlights）
	│      2.Highlight repair(Dynamic threshold percentage+NS).ipynb（Percentage dynamic threshold +NS technique fixes the code for highlights）
	│      2.Highlight repair(Dynamic threshold statistics+FMM).ipynb.ipynb（Code with statistical dynamic thresholding +FMM technique for fixing highlights）
	│      2.Highlight repair(Fixed threshold+FMM).ipynb（Code with fixed threshold +FMM technique to fix highlights）
	│      2.Highlight repair(Fixed threshold+NS).ipynb（Fixed threshold +NS technique fixes the code for highlights）
	│
	└─3.Image channel（Image channel conversion techniques）
	        3.Image channel transformation(RGB to CMY).ipynb（RGB image channel to CMY image channel code）
	        3.Image channel transformation(RGB to HSI).ipynb（RGB image channel to HSI image channel code）
	        3.Image channel transformation(RGB to YIQ).ipynb（RGB image channel to YIQ image channel code）

## 3.Version notes for python packages:

Windows：environment-windows.yml

Linux: segnet-vgg.ipynb and segnet.ipynb is linux environment, related python packages are: environment-linux.yml

# 息肉分割

## 		1.数据相关说明：

本研究的训练集和测试集以及模型相关权重（300轮以内的最佳权重）可以在该链接下载：https://drive.google.com/drive/folders/1scOMQk3PPxkKcLRNLMJrHR_1NpZOsvRV?usp=sharing

你可以使用Sample-dataset.h5中的简单数据跑通代码

## 		2.文件相关说明：

以下是code文件夹目录下文件和说明：

	│  How to use h5 files.ipynb  （h5文件使用方法）
	|  Sample-dataset.h5（样例数据）
	│  The random numbers partition Train and Val.ipynb（从训练集划分验证集）
	│  Train.txt（划分出的训练集）
	│  Val.txt（划分出的验证集）
	│
	├─Deep learning Mode（深度学习模型文件夹）
	│      FPN+V3.ipynb（FPN+V3模型代码）
	│      Linknet+V3.ipynb（Linknet+V3模型代码）
	│      PraNet.ipynb（PraNet模型代码）
	│      PSPnet+V3.ipynb（PSPnet+V3模型代码）
	│      SegNet-Simple.ipynb（简化SegNet模型代码）
	│      SegNet.ipynb（SegNet模型代码）
	│      U-net++.ipynb（U-net++模型代码）
	│      U-net+V3.ipynb（U-net+V3模型代码）
	│
	└─Digital Image Methods（数字图像增强技术方法文件夹）
	├─1.Edge detection（边缘检测技术）
	│      1. img+edge Enhancing images.ipynb（边缘图像+原始图像得到增强后图像的代码）
	│      1.BHPF (Extract edges in batches).ipynb（巴特沃斯高通滤波器获取边缘图像的代码）
	│      1.Canny (Extract edges in batches).ipynb（Canny算子获取边缘图像的代码）
	│      1.GHPF (Extract edges in batches).ipynb（高斯高通滤波器获取边缘图像的代码）
	│      1.IHPF (Extract edges in batches).ipynb（理想高通滤波器算子获取边缘图像的代码）
	│      1.Marr (Extract edges in batches).ipynb（Marr 算子获取边缘图像的代码）
	│      1.Prewitt (Extract edges in batches).ipynb（Prewitt 算子获取边缘图像的代码）
	│      1.Roberts (Extract edges in batches).ipynb（Roberts 算子获取边缘图像的代码）
	│      1.Sobel (Extract edges in batches).ipynb（Sobel 算子获取边缘图像的代码）
	│
	├─2.Highlight repair（高光修复技术）
	│      2.Highlight repair(Dynamic threshold percentage+FMM).ipynb（百分比动态阈值+FMM技术修复高光的代码）
	│      2.Highlight repair(Dynamic threshold percentage+NS).ipynb（百分比动态阈值+NS技术修复高光的代码）
	│      2.Highlight repair(Dynamic threshold statistics+FMM).ipynb.ipynb（统计动态阈值+FMM技术修复高光的代码）
	│      2.Highlight repair(Fixed threshold+FMM).ipynb（固定阈值+FMM技术修复高光的代码）
	│      2.Highlight repair(Fixed threshold+NS).ipynb（固定阈值+NS技术修复高光的代码）
	│
	└─3.Image channel（图像通道转化技术）
	        3.Image channel transformation(RGB to CMY).ipynb（RGB图像通道转CMY图像通道的代码）
	        3.Image channel transformation(RGB to HSI).ipynb（RGB图像通道转HSI图像通道的代码）
	        3.Image channel transformation(RGB to YIQ).ipynb（RGB图像通道转YIQ图像通道的代码）

## 3.相关python包的版本说明：

Windows：environment-windows.yml

Linux：其中SegNet-Simple.ipynb和SegNet.ipynb为linux环境，相关python包为：environment-linux.yml
