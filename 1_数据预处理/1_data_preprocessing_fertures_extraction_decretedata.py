# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 19:25:01 2018

@author: a
"""

#%%==============特征数据预处理-离散值处理===========
"""
#预读笔记

-关于标签label的处理：（“标签处理”）
    如果得到的数据，Y值是分类变量，首先要转换成数值型0,1,2等来表示的话，结合set(),map(dict)实现
    set(class_label):class_label有几类，则返回每一类的值
    map(dict):dict建立映射，DataFrame通过map函数实现数据转换，如dict={A:1},dataframe['v'] = dataframe['v'].map(dict)，原先v列的A就会转换成1

-关于特征feature的处理：（“特征处理”，“map反变换”）
    1.同上，利用map映射
    2.map反变换的实现：根据之前的mapping构建反变换的映射（{v:k for k , v in color_mapping.items()}）

#####
-scikit LabelEncoder:作用同上标签处理，将分类变量转换成整数数据，如class1，class2，...转换成0，1，...
    #le = LabelEncoder()
    #df['class label'] = le.fit_transform(df['class label'])
    #df['class label'] = le.inverse_transform(df['class label'])#反变换

-df.transpose().to_dict().values()
    #df.transpose():实现dataframe转置
    #dict.to_dict():返回一个dataframe，按列索引的，如{0：{A:1,B:2,C:3},...} 这里0表示第一列，：后面则是这一列对应的字典
                    由于原来的dataframe每一行对应一个样本，因此df.transpose().to_dict()得到的应该是每个样本（即第i=0,1,2,3个样本）对应的各个变量的字典
                    具体实现在后面的代码中


-scikit DictVectorizer
    #df.iloc[:,:,-1]:去掉最后一列；iloc可以实现按位置选择，(integer-location based indexing for selection by position)
    #DictVectorizer.fit_transform():看后面代码，感觉不是很好用 


-OneHotEncoder
    #先将分类数据转换成整数数据，才能使用OneHotVectorizer
    #df['color'] = LabelEncoder().fit_transform(df['color'])#转换成整数
    #X = OneHotEncoder(sparse = False).fit_transform(df[['color']].values)
    #最终返回的是df['color']这一列对应的OneHot矩阵（每一行就是OneHot向量，即各个样本在color变量上的表示）


-pandas.get_dummies
    #直接看后面代码，利用get_dummies(df)就可以得到整个dataframe的哑变量形式，
    #特别的，如果只有color列是分类变量的形式，而其他变量都是整数型数据，那么哑变量的处理只会对color这一列进行处理


-df['color']与df[['color']]：
    #要注意两者的区别,前者返回的是Series数据，后者返回的是dataframe数据
    #要单独取指定的某一列出来，得到新的dataframe，则用后者，比如df[['color','size']],这里就得到color和size两列组成的新的dataframe
    #前者则是得到一组序列数据，如果df['color'].values,则可以看到结果是一维数组的形式，而如果df[['color']].values,则可以看到结果是color这一列各个值
"""



#%%   1.
import pandas as pd
df = pd.DataFrame(  [
                    ['green', 'M',  10.1, 'class1'],
                    ['red',   'L',  13.5, 'class2'],
                    ['blue',  'XL', 15.3, 'class3'],
                    ['red',   'L',  12.5, 'class2']
                    ]) 
df.columns = ['color','size','price','class label']

df

#%%   2.标签处理
#2.1 通常把字符型标签转换成数值型
"""#
class_label = df['class label']
set(class_label)#set() : 重复无序的值
"""
class_mapping = {label:idx for idx , label in enumerate(set(df['class label']))}
#class_mapping : dict
df['class label'] = df['class label'].map(class_mapping)#pandas.DataFrame.map(dict):利用映射进行数据转换

df

#%%  3.特征处理
size_mapping = {
            'XL':3,
            'L' :2,
            'M' :1}
df['size'] = df['size'].map(size_mapping)
#df

color_mapping = {
            'green':(0,0,1),
            'red'  :(0,1,0),
            'blue' :(1,0,0)}
df['color'] = df['color'].map(color_mapping)
df

#%%  4.map反变换
#color_mapping.items()
inv_color_mapping = {v:k for k , v in color_mapping.items()}
inv_size_mapping  = {v:k for k , v in size_mapping.items()}
inv_class_mapping = {v:k for k , v in class_mapping.items()}

df['color'] = df['color'].map(inv_color_mapping)
df['size']  = df['size'].map(inv_size_mapping)
df['class label'] = df['class label'].map(inv_class_mapping)
df





#%% ===============Using scikit-learn and pandas features

#%%  1.scikit LabelEncoder
from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()
#df
df['class label'] = class_le.fit_transform(df['class label'])
#df
"""
#反变换回去可以用函数：inverse_transform()
df['class label'] = class_le.inverse_transform(df['class label'])
df
"""

#%%  2.scikit DictVectorizer
df.transpose().to_dict().values()
"""
#df.transpose():转置
#df.transpose().to_dict():按列索引返回一个字典
"""

feature = df.iloc[:, :-1]#去掉最后一列（分类）
#df.iloc[]:可以实现按照行列号访问返回dataframe
feature

###
from sklearn.feature_extraction import DictVectorizer

dvec = DictVectorizer(sparse= False)
X = dvec.fit_transform(feature.transpose().to_dict().values())
X


#可以调用get_feature_names()来返回新的列的名字，其中0和1就代表是不是这个属性
pd.DataFrame(X, columns=dvec.get_feature_names())


#%%  3. OneHotEncoder
#OneHotEncoder 必须使用整数作为输入，所以得预处理
df

color_le = LabelEncoder()
df['color'] = color_le.fit_transform(df['color'])
#df


##
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse = False)

X = ohe.fit_transform(df[['color']].values)
X

"""
df[['color']].values
df['color'].values
"""
#
df['color'] = color_le.inverse_transform(df['color'])
df
#%%  4.pandas.get_dummies
#Pandas库中同样有类似的操作，使用get_dummise也可以得到相应的特征
import pandas as pd
df = pd.DataFrame(  [
                    ['green', 'M',  10.1, 'class1'],
                    ['red',   'L',  13.5, 'class2'],
                    ['blue',  'XL', 15.3, 'class3'],
                    ['red',   'L',  12.5, 'class2']
                    ]) 
df.columns = ['color','size','price','class label']
df

###
size_mapping = {'XL':3,'L':2,'M':1}
df['size'] = df['size'].map(size_mapping)

class_mapping = {label:idx for idx,label in enumerate(set(df['class label']))}
df['class label'] = df['class label'].map(class_mapping)

df
###对整个DF使用get_dummise将会得到新的列（哑变量）
pd.get_dummies(df)#由于此时df只有一列是分类变量，故得到的新的列只与color列有关


