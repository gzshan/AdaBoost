#coding=utf-8
import os
from matplotlib import pyplot as plt
import numpy as np

"""为了实现AdaBoost算法，首先构造一个简单的数据集来测试"""
def loadSimpData():
	dataMat = np.matrix([[1.,2.1],[2.,1.1],[1.3,1.],[1.,1.],[2.,1.]])
	classLabels = [1.0,1.0,-1.0,-1.0,1.0]
	return dataMat,classLabels

"""更为一般的数据加载方法，这里数据文件格式为：前n-1列为属性，最后一列为其对应的标签"""
def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t')) #属性的数目
	dataMat = []
	labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat-1):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat,labelMat


#通过阈值对数据进行分类，其实就是判断大于阈值的为-1还是小于阈值的为-1
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq): #dimen代表哪一维，也就是哪一个属性
	retArray = np.ones((np.shape(dataMatrix)[0],1)) #m*1
	#print(retArray)
	if threshIneq == 'lt':
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:,dimen] > threshVal] = -1.0
	return retArray

#建立决策树桩，训练集，对应的标签，各个样本的权重D
#决策树的简化版本，相对应于adoboost算法中的基础学习器
def buildStump(dataArr,classLabels,D):
	dataMatrix = np.mat(dataArr); #训练集，每一行为一个样本，每一列为一个属性
	labelMat = np.mat(classLabels).T #每一个样本对应的标签
	m,n = np.shape(dataMatrix) # m为样本的个数，n为属性的个数

	numSteps = 10.0
	bestStump = {} #最优决策树桩
	bestClasEst = np.mat(np.zeros((m,1))) #保存此学习器给出的分类结果
	minError = np.inf #最小误差
	
	"""对每一个属性进行遍历"""
	for i in range(n):
		#决策树对连续值属性的处理
		rangeMin = dataMatrix[:,i].min()
		rangeMax = dataMatrix[:,i].max()
		stepSize = (rangeMax-rangeMin)/numSteps #每一步的步长

		for j in range(-1,int(numSteps)+1): #依次判断连续值的每一个候选划分点
			for inequal in ['lt','gt']: #根据大于还是小于阈值，判断哪一边为-1，标准是误差最小
				threshVal = (rangeMin+float(j)*stepSize) 
				predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal) #根据阈值进行分类
				errArr = np.mat(np.ones((m,1)))
				errArr[predictedVals==labelMat] = 0
				weightedError = D.T * errArr #总的错误率
				#print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
				if weightedError < minError:  #根据错误率找出最佳决策树桩
					minError = weightedError
					bestClasEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	return bestStump,minError,bestClasEst

#基于决策树桩实现完整的adaboost算法,注意：这里标签应该设置为+1和-1
def adaboostTrainDS(dataArr,classLabels,numIt=40): #dataArr样本矩阵，m行(m个样本)，n列（n个属性），classLabels：1*m，m个标签
	weakClassArr=[] #保存所有的基分类器
	m = np.shape(dataArr)[0] #样本个数
	D = np.mat(np.ones((m,1))/m) #样本权重矩阵，m*1，初始都为1/m
	aggClassEst = np.mat(np.zeros((m,1))) #m*1的矩阵，保存最后结合之后的最终分类结果，结合方法：加权平均法

	"""迭代执行算法直到到达指定的基学习器数量，或者样本分类全部正确"""
	for i in range(numIt):
		"""第一步：基于当前分布D训练出一个基学习器（这里是简单的决策树桩）"""
		bestStump,error,classEst = buildStump(dataArr,classLabels,D) #返回的依次是：分类器，此学习器的错误率，此学习器给出的分类结果
		#print("D:",D.T) #打印当前权重

		"""第二步：分类器的权重更新公式，alpha=0.5ln(1-error/error)"""
		alpha = float(0.5*np.log((1.0-error)/max(error,1e-16))) 
		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump)
		#print("classEst: ",classEst.T)

		"""第三步：更新每一个样本的权重，使得正确分类的样本权重降低而错分样本权重升高"""
		expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst) #classLables（真实值）：1*m  classEst（预测值）： m*1
		D = np.multiply(D,np.exp(expon)) #样本权重更新公式
		D = D/D.sum()

		"""第四步：将之前得到的学习器进行结合，加权平均法"""
		aggClassEst += alpha*classEst
		#print("aggClassEst:",np.sign(aggClassEst.T)) #输出集成之后的分类结果

		"""计算集成之后的错误率"""
		aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
		errorRate = aggErrors.sum()/m
		print("total error: ",errorRate)
		print('\n')

		#样本分类全部正确，算法迭代提前终止
		if errorRate == 0.0: 
			break

	return weakClassArr,aggClassEst

"""测试代码，对于一个新的测试样本，给出其判定结果"""
def adaClassify(dataToClass,classifierArr):
	dataMatrix = np.mat(dataToClass)
	m = np.shape(dataMatrix)[0]
	aggClassEst = np.mat(np.zeros((m,1))) #保存最终的集成加权平均结果
	for i in range(len(classifierArr)):
		"""得到在每一个分类器上的结果"""
		classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
		aggClassEst += classifierArr[i]['alpha']*classEst #加权平均
		#print(aggClassEst)
	return np.sign(aggClassEst)


if __name__ == '__main__':
	dataMat,classLabels = loadSimpData()
	print("训练集：",dataMat)
	print("标签：",classLabels)
	print("\n")
	#plt.scatter(dataMat[:,0].tolist(),dataMat[:,1].tolist())
	#plt.xlim(0.8,2.2)
	#plt.ylim(0.8,2.2)
	#plt.show()

	#D = np.mat(np.ones((5,1))/5)
	#bestStump,minError,bestClasEst=buildStump(dataMat,classLabels,D)
	#print(bestStump,minError,bestClasEst)
	classifierArray,aggClassEst = adaboostTrainDS(dataMat,classLabels,9)
	print(classifierArray,aggClassEst)
	result = adaClassify([[5,5],[0,0]],classifierArray)
	print(result)
	dataArr,labelArr = loadDataSet('horseColicTraining2.txt')
	classifierArray,aggClassEst = adaboostTrainDS(dataArr,labelArr,10)
	
