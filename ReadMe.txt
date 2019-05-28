1.input文件夹中是本次实验的数据集，共六个数据集
2.output文件夹中有suprised和unsuprised两个子文件夹，其中suprised文件夹中是监督模型得到的六个数据集各自的Popt和ACC以及一个总的Popt和ACC
3.suprised_model里面是具体的监督模型的代码，unsuprised_model中是无监督模型的代码
	suprised_model
		util.py：里面是辅助函数
		suprised.py:里面是算法流程
		main_one.py:是一个项目bugzilla的实验,这个文件就类似于c语言中的main函数一样，是主要的控制流程。
		main_many.py:是全部项目的实验
		corr_var.py：是计算项目中的属性的相关性，方差等，是特征工程的一部分，但是这里因为是重复论文，论文中已经告诉我们哪些属性是高度相关的属性了，所以其实用不到这一块的，这是我自己验证的。
		
		resultOutput.py：是打印有监督模型得到的结果的盒图的。
4.resultOutput.py：是打印有监督和无监督模型的盒图。
5.基于代际敏感的即使缺陷预测——有监督和无监督模型对比：是对整个代码的说明。




