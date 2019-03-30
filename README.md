# 研究生《模式识别实验》代码及分析

## EXP1   极大似然估计和线性判别分析
***MLE.py:***  
高斯分布数据数据集的极大似然估计。  

***Fisher.py:***  
Fisher LDA，并画出投影示意图。


## EXP2   感知器算法及各种变种
***SGDAlgs.py:***               
类文件。类似于sklearn中的SGDClassifier和SGDRegressor类，使用SGD优化算法。选择平移的hinge loss为感知器，选择hinge loss加L2正则化为SVM，选择log loss为Logistic Regression，选择mse loss为线性回归。  

***ErrorCorrectingAlgs.py:***   
类文件。实现了各种感知器的变种。抽象程度低于SGDAlgs，是一个副产品。例如LR和线性回归就无法纳入此框架。  

***linear_discriminator.py:***  
函数文件。将ErrorCorrectingAlgs中的算法拆分，是一个副产品。可忽略。  

***per_and_svm:***              
测试文件。单独比较感知器和SVM，验证两者关系。  

***test_clf:***                 
测试文件。测试SGDAlgs和ErrorCorrectingAlgs中实现的各种算法。  

***文件顺序***
总体来说，首先编写了linear_discriminator.py，每个函数各自实现了一种感知器变种，存在大量重复代码。因此创建了ErrorCorrectingAlgs类，即误差校正算法，将各种感知器变种纳入一个框架，使用共同的训练和预测函数，仅梯度求解部分需要分别实现。随后推广为SGDAlgs算法，将感知器变种，SVM，Logistic Regression和Linear Regression等算法统一在一起。


## EXP3   验证方差-偏差分解
***BV.py:***    
用多项式回归验证方差-偏差分解，并画出示意图。


## EXP4   没有免费的午餐
***no_free_lunch.py:***   
用K近邻分类器解释没有免费的午餐定理。


## EXP5   K-means和模糊C-means
***KMeans.py:***    
K-mean算法。    

***FCM.py:***       
模糊C-Means算法(FCM)。  

***test.py:***      
测试效果并画图。


## EXP6   降维算法
***DimensionReduction.py:***  
实现了PCA和Multi-class LDA。  

***test.py:***                
测试效果。  

