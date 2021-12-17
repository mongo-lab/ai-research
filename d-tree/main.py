import pandas as pd
from random import randint
import numpy as np
import treeplotter
from collections import Counter
import math
import matplotlib.pyplot as plt

#from sklearn.tree import DecisionTreeRegressor
#from sklearn.model_selection import train_test_split
''' def play():
    random_int = randint(0,100)

    while True:
        user_guess = int (input("what number did we guess(0-100)?"))
        if user_guess == random_int:
            print("you found the number.Congrats!")
            break
        if user_guess < random_int:
            print("your number is less than the number we guessed.")
            continue
        if user_guess > random_int:
            print("your number is more than the number we guessed.")
            continue
'''
    #def DecisionTree():
#分割数据集
def split_train(data,test_ratio):
        shuffled_indices = np.random.permutation(len(data))
        test_set_size = int(len(data)*test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices=shuffled_indices[test_set_size:]
        return data.iloc[train_indices],data.iloc[test_indices]

#计算信息熵
def calcShannonEnt(dataSet):
        numEntries = len(dataSet) #样本数
        labelcounts = {} #key是目标分类的类别，value是属于该类别的样本个数
        for featVec in dataSet:#遍历整个数据集，每次取一行
                currentLabel = featVec[-1] #取该行最后一列的值
                if currentLabel not in  labelcounts.keys():labelcounts[currentLabel] = 0
                labelcounts[currentLabel] +=1
        shannonEnt = 0.0 #初始化信息熵
        for key in labelcounts:
                prob = float(labelcounts[key])/numEntries
                shannonEnt -= prob * math.log(prob,2) #计算信息熵
        return shannonEnt
#按给定的特征划分数据
def splitDataSet(dataSet,axis,value):#axis是数据集dataSet要进行特征划分的列号，value是该列下某个特征值
        retDataSet = []
        for featVec in dataSet:#遍历数据集，并抽取按axis的当前value特征进行划分的数据集（不包含axis列的值）
                if featVec[axis] == value:
                        reducedFeatVec = featVec[:axis]
                        reducedFeatVec += featVec[axis+1:]
                        retDataSet.append(reducedFeatVec)
                        #print axis,value,reducedFeatVec
        #print retDataSet
        return retDataSet

#选取当前数据集下，用于划分数据集的最优特征
def chooseBestFeatureToSplit(dataSet):
        numFeatures = len(dataSet[:0]) - 1 #获取当前数据集的特征个数，最后一列是分类标签
        baseEntropy = calcShannonEnt(dataSet) #计算当前数据集的信息熵
        bestInfoGain = 0.0;bestFeature = -1 #初始化最优信息在增益和最优特征
        for i in range(numFeatures): #遍历每个特征
                featList = [exampe[i] for example in dataSet] #获取数据集中当前特征下的所有值
                uniqueVals = set(featList) #获取当前特征值
                newEntropy = 0.0
                for value in uniqueVals: #计算每种划分方式的信息熵
                        subDataSet = splitDataSet(dataSet,i,value)
                        prob = len(subDataSet)/float(len(dataSet))
                        newEntropy += prob * calcShannonEnt(subDataSet)
                infoGain = baseEntropy - newEntropy #计算信息增益
                if(infoGain > bestInfoGain):    #比较每个特征的信息增益，选择最好的
                        bestInfoGain = infoGain
                        bestFeature = i
        return bestFeature

#出现次数最多的分类名称
def majorityCnt(classList):
        classCount = {}
        for vote in classList:
                if vote not in classCount.keys():classCount[vote] = 0
                classCount[vote] += 1
        sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse = True)
        return sortedClassCount[0][0]

#生成决策树
def createTree(dataSet,labels):
        print(dataSet)
        classList = [example[-1] for example in dataSet] #返回当前数据集下标签列所有值
        print(classList)
        if classList.count(classList[0]) == len(dataSet):
                return classList[0] #当类别完全相同时则停止继续划分，直接返回该类的标签
        if len(dataSet[:0]) == 1:#遍历完所有的特征时，仍不能将数据集划分成仅包含唯一类别的分组
                return majorityCnt(classList) #由于无法简单的返回唯一的类标签，这里就返回出现次数最多的类别作为返回值
        bestFeat = chooseBestFeatureToSplit(dataSet)
        bestFeatLabel = labels[bestFeat]
        #绘制树
        myTree = {bestFeatLabel:{}}
        del(labels[bestFeat]) #删除已经在选取的特征
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
                subLabels = labels[:]
                myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
        return myTree


def createPlot(inTree):
        fig = plt.figure(1, facecolor='white')
        fig.clf()
        axprops = {'xticks': None, 'yticks': None}
        createPlot.ax1 = plt.subplot(111, frameon=False)
        plotTree.totalW = float(getNumLeafs(inTree))  # 全局变量宽度 = 叶子数目
        plotTree.totalD = float(getTreeDepth(inTree))  # 全局变量高度 = 深度
        plotTree.xOff = -0.5 / plotTree.totalW
        plotTree.yOff = 1.0
        plotTree(inTree, (0.5, 1.0), '')
        plt.show()

def plotTree(myTree, parentPt, nodeTxt):
        numLeafs = getNumLeafs(myTree)
        depth = getTreeDepth(myTree)
        firstStr = list(myTree.keys())[0]
        # cntrPt文本中心点， parentPt指向文本中心的点
        cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
        plotMidText(cntrPt, parentPt, nodeTxt)
        plotNode(firstStr, cntrPt, parentPt, descisionNode)
        seconDict = myTree[firstStr]
        plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
        for key in seconDict.keys():
                if type(seconDict[key]).__name__ == 'dict':
                        plotTree(seconDict[key], cntrPt, str(key))
                else:
                        plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
                        plotNode(seconDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
                        plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
        plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def plotMidText(cntrPt, parentPt, txtString):
        xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
        yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
        createPlot.ax1.text(xMid, yMid, txtString, va='center', ha='center', rotation=30)

def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]     # 这个是改的地方，原来myTree.keys()返回的是dict_keys类，不是列表，运行会报错。有好几个地方这样
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

descisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    # nodeTxt为要显示的文本，centerNode为文本中心点, nodeType为箭头所在的点， parentPt为指向文本的点
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                             xytext=centerPt, textcoords='axes fraction',
                              va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)

'''def job_type(s):
        it = {'admin':1,'technician':2,'services':3,'management':4,'retired':5,'blue-collar':6}
        return it[s]

def marital_type(s):
        it = {'married':1,'single':2,'divorced':3}
        return it[s]

def housing_type(s):
        it = {'yes':1,'no':2}
        return it[s]'''


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
        bank_file_path = 'bank.csv'
        data_type = {'acoustic_data':np.int16,'time_to_failure':np.float64}
        bank_data = pd.read_csv(bank_file_path,dtype = data_type)

        #print(bank_data.columns)
        bank_data = bank_data.dropna(axis=0)
        y = bank_data.deposit
        bank_features = [ 'job', 'marital', 'education','deposit']
        X = bank_data[bank_features]
        #print(X.head(20))

        #划分训练集和测试集
        train,val= split_train(X,00.5)
        train_x = train[bank_features]
        train_y = train.deposit
        val_x = val[bank_features]
        val_y = val.deposit
        train_data = np.array(train)
        train_data_list = train_data.tolist()
        lensesTree = createTree(train_data_list,bank_features)
        print(lensesTree)
        createPlot(lensesTree)
        #train_y = train.deposit
        #train_X = train[bank_features]
        #predict_y = train.deposit
        #predict_X= train[bank_features]

       #sklearn
  #      melbourne_model = DecisionTreeRegressor(random_state=1)
 #       melbourne_model.fit(train_x, train_y)
        #melbourne_model = DecisionTreeRegressor()
       # melbourne_model.fit(train_X, train_y)
        #val_predictions = melbourne_model.predict(val_X)
  #      createPlot(melbourne_model.predict(val_x))
        #print(train)
        #print(train_X)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
