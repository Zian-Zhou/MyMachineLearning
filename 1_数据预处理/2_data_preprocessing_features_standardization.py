# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 13:22:26 2018

@author: a
"""

#%%==================特征数据预处理-标准化与归一化======================
"""
sklearn preprocessing

-preprocessing.StandardScale().fit(dataframe[['...','....']]).transform(dataframe[['...','....']])
-preprocessing.MinMaxScaler().fit(dataframe[['...','....']]).transform(dataframe[['...','....']])

-自定义的两个作图函数，作图非常好看，可以作为模板:plot_compare_transform,plot_compare_PCA
#plot_compare_transform.png
#plot_compare_PCA.png
"""


#%% loading the wine dataset
import pandas as pd
import numpy as np

df = pd.read_csv('wine_data.csv',usecols=[0,1,2])
df.columns=['class label','alcohol','malic acid']
df.head()#看头几行

#数据中，alcohol和malic acid衡量的标准不一样，特征之间数值差异较大

#%%  standardization and min-max scaling
"""
-standardization:z = (x-mean)/std
-min-max scaling = (x-x_min)/(x_max - x_min)
"""

from sklearn import preprocessing

####transform
#1.standardscale
std_scale = preprocessing.StandardScaler().fit(df[['alcohol','malic acid']])
df_std = std_scale.transform(df[['alcohol','malic acid']])
#2.min-max
minmax_scale = preprocessing.MinMaxScaler().fit(df[['alcohol','malic acid']])
df_minmax = minmax_scale.transform(df[['alcohol','malic acid']])

#show the data after tranformation
#Mean and Standard deviation after standardization
print(' Mean after standardization:\n alcohol={:,.2f},malic acid = {:,.2f}'.format(df_std[:,0].mean(),df_std[:,1].mean()))
print('\n Standard deviation after standardization:\n alcohol={:,.2f},malic acid={:,.2f}'.format(df_std[:,0].std(),df_std[:,1].std()))

#Max and Min deviation after min-max scaling
print(' Min after min-max scaling:\n alcohol={:,.2f},malic acid = {:,.2f}'.format(df_minmax[:,0].min(),df_minmax[:,1].min()))
print('\n Max after min-max scaling:\n alcohol={:,.2f},malic acid={:,.2f}'.format(df_minmax[:,0].max(),df_minmax[:,1].max()))




#%% Plotting
%matplotlib inline

from matplotlib import pyplot as plt
###图像很好看！！！！
def plot_compare_transform():
    ###
    """
    将原始的数据图和变换后的图放在了同个图上
    """
    plt.figure(figsize=(8,6))
    
    plt.scatter(df['alcohol'],df['malic acid'],color='green',label='input scale',alpha = 0.5)
    
    plt.scatter(df_std[:,0],df_std[:,1],color='red',label='Standardization [$N  (\mu=0, \; \sigma=1)$]',alpha = 0.3)
    
    plt.scatter(df_minmax[:,0],df_minmax[:,1],color='blue',label='min-max scaled [min=0, max=1]',alpha=0.3)
    
    plt.title('Alcohol and Malic Acid content of the wine dataset')
    plt.xlabel('alcohol')
    plt.ylabel('malic acid')
    plt.legend(loc='upper left')
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    ###
    
plot_compare_transform()




#%% 作图观察数据有无因为变换而改变原来的分布信息
fig, ax = plt.subplots(3,figsize=(6,14))

for a,d,l in zip(range(len(ax)),
                 (df[['alcohol','malic acid']].values,df_std,df_minmax),
                 ('Input scale',
                  'standardization [$N  (\mu=0, \; \sigma=1)$]',
                  'min-max scaled [min=0, max=1]')
                 ):
                     for i,c in zip(range(1,4),('red','blue','green')):
                         ax[a].scatter(d[df['class label'].values == i,0],
                                       d[df['class label'].values == i,1],
                                       alpha=0.5,
                                       color=c,
                                       label='Class %s' %i)
                         
                     ax[a].set_title(l)
                     ax[a].set_xlabel('alcohol')
                     ax[a].set_ylabel('malic acid')
                     ax[a].legend(loc='upper left')
                     ax[a].grid()
                     
plt.tight_layout()
plt.show()

#%% 注意
"""
在机器学习中，如果我们对训练集做了上述处理，那么同样的对测试集也必须要经过相同的处理
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train = std_scale.transform(X_train)
X_test = std_scale.transform(X_test)
"""




#%%  ------------------探索标准化处理对PCA主成分分析的影响-------------
"""
主成分分析PCA是一个非常有用的套路，接下来，咱们来看看数据经过标准化处理和未标准化处理后使用PCA
"""
import pandas as pd 
df = pd.read_csv('wine_data.csv')
df.head()

#%%
from sklearn.model_selection import train_test_split

X_wine = df.values[:,1:]
y_wine = df.values[:,0]

X_train, X_test, y_train, y_test = train_test_split(X_wine,y_wine,test_size=0.30, random_state=12345)

#标准化
from sklearn import preprocessing
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

#%%   使用PCA进行降维，将数据集转换成二维特征子空间
from sklearn.decomposition import PCA

###未进行标准化
pca = PCA(n_components=2).fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

###进行了标准化的PCA
pca_std = PCA(n_components=2).fit(X_train_std)
X_train_std = pca_std.transform(X_train_std)
X_test_std = pca_std.transform(X_test_std)



#%% 效果展示对比
%matplotlib inline

from matplotlib import pyplot as plt
def plot_compare_PCA():
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))#figsize:控制每个图的大小

    for l,c,m in zip(range(1,4), ('blue', 'red', 'green'), ('^', 's', 'o')):
        ax1.scatter(X_train[y_train==l, 0], X_train[y_train==l, 1],
                    color=c,
                    label='class %s' %l,
                    alpha=0.5,
                    marker=m
                    )

    for l,c,m in zip(range(1,4),('blue','red','green'),('^','s','o')):
        ax2.scatter(X_train_std[y_train==l, 0], X_train_std[y_train==l, 1],
                    color=c,
                    label='class %s' %l,
                    alpha=0.5,
                    marker=m
                    )

    ax1.set_title('Transform NON-standardizated training dataset after PCA')
    ax2.set_title('Transform standardizated training dataset after PCA')

    for ax in (ax1,ax2):
        ax.set_xlabel('1st principal component')
        ax.set_ylabel('2nd principal component')
        ax.legend(loc='upper right')
        ax.grid()
    
    plt.tight_layout()
    plt.show()

plot_compare_PCA()

"""
plot_compare_PCA.jpg
作图可以看出，经过标准化的数据可分性更强
"""

