#%% -----------------------------------
"""
機械学習（信号解析で信号特徴が判明した後の自動判別化）
(1)信号解析で異なる特徴があることを把握
(2)信号特徴とその結果の組み合わせを機械学習器に覚えこませる
(3)信号のみを学習済みの機械学習器に入力し，結果を予測させる
"""

#リセット用ライブラリ
from IPython import get_ipython 
#リセット
get_ipython().magic('reset -sf')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm #機械学習ライブラリ
from sklearn.metrics import accuracy_score, confusion_matrix #ベクターマシン関連
import seaborn as sn #解析用

plt.close('all')

#%% -----------------------------------
"""
学習データの作成：
3種類の状態（状態１，状態２，状態３）となった時のセンサデータ10変数と，状態ラベルのセット
"""

# 1種類のデータセット数
num = 40

# 入力部分：（1データは1行10列で1種類40データ（各データは10変数）の集合体）
# 時系列データをそのまま入力
condition1 = np.random.randint( 0, 10,(num,10)) # 状態１×40データセット
condition2 = np.random.randint(10, 20,(num,10)) # 状態２×40データセット
condition3 = np.random.randint(20, 30,(num,10)) # 状態３×40データセット

# 理想的な出力部分：1種類目「1」，2種類目「2」，3種類目「3」
target1 = np.full(num, 1) #最終列にラベルを置く
target2 = np.full(num, 2)
target3 = np.full(num, 3)

#わざと，データの間違えを作り出すため，ラベル「1」であるものラベル「2」とする．
#ラベル「1」のデータは39セット，ラベル「2」は41セット，ラベル「3」は40セット
#target1[0] = 2 #ダミーデータ

# 上記のデータを統合して学習データとする
learning_data_all = np.vstack([condition1, condition2, condition3])
target_all = np.hstack([target1, target2, target3]) 

#%%
#-------------------------------------
# 学習用データのファイル保存
#-------------------------------------
learning_data_set=np.hstack([learning_data_all,np.array([target_all]).T])

# ファイル保存
df = pd.DataFrame(learning_data_set)
df.to_csv('learning_data_set1.csv', index=False, header=False)

#%% -----------------------------------

plt.close('all')
plt.figure(1)
plt.rcParams['figure.figsize'] = (3.0, 9.0) #グラフのサイズ設定
plt.rcParams['font.size'] = 11     #全体のフォントサイズ

plt.subplot(3,2,1)
plt.xlim(0,10)
plt.ylim(0,30)
plt.ylabel('Value')
plt.xlabel('Parameter')
plt.title('Condition1') 
plt.grid()

plt.subplot(3,2,3)
plt.xlim(0,10)
plt.ylim(0,30)
plt.ylabel('Value')
plt.xlabel('Parameter')
plt.title('Condition2')
plt.grid()

plt.subplot(3,2,5)
plt.xlim(0,10)
plt.ylim(0,30)
plt.ylabel('Value')
plt.xlabel('Parameter')
plt.title('Condition3')
plt.grid()

for i in range(0,len(target_all)):

    if target_all[i]==1:
        plt.subplot(3,2,1)  # 良品1      
        plt.plot(learning_data_all[i,:].T,label=str(i))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, ncol=5)

    elif target_all[i]==2:
        plt.subplot(3,2,3) # 良品2
        plt.plot(learning_data_all[i,:].T,label=str(i))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, ncol=5)

    elif target_all[i]==3:
        plt.subplot(3,2,5) # 不良品3
        plt.plot(learning_data_all[i,:].T,label=str(i))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, ncol=5)

plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
#plt.tight_layout()
plt.show()

#%% -----------------------------------
"""
SVMを用いて分類を行う。
非線形SVMで2次元の多項式カーネルを使用する。　ここのブロックで入出力の傾向をAIに解析させる
"""

# poly: 多項式カーネル　線形とかのがある
# degree: 多項式カーネル関数の次数 2にしておけばだいたい大丈夫
clf = svm.SVC(kernel='poly', degree=2)
clf.fit(learning_data_all, target_all)

#%%
# トレーニングデータに対する精度
pred_train = clf.predict(learning_data_all)
accuracy_train = accuracy_score(target_all, pred_train)
print('トレーニングデータに対する正解率： %.3f' % accuracy_train)

#------------------グラフ--------------------
# グラフ2
plt.figure(2)
plt.rcParams['figure.figsize'] = (8.0, 8.0) #グラフのサイズ設定
plt.rcParams['font.size'] = 11     #全体のフォントサイズ

X0, X1 = learning_data_all[:, 0], learning_data_all[:, 1]
plt.scatter(X0, X1, c=target_all, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.xlabel('column_1')
plt.ylabel('column_2')
plt.title('SVC with polynomial (degree 2) kernel')
plt.show()

# グラフ3
plt.figure(3) #混合行列　おぼえこませたデータが勘違いなく学習できているかのっ指標になる　非常に重要
plt.rcParams['figure.figsize'] = (8.0, 8.0) #グラフのサイズ設定
plt.rcParams['font.size'] = 11     #全体のフォントサイズ # 今回は対角線上に40,40,40となっていればよい
# 混合行列の見方　横ライン->各出力の学習における総データ数　縦ライン->各データに対応する各出力　詳細はteamsの資料参照
labels = sorted(list(set(target_all.astype('int64'))))
cmx_data = confusion_matrix(target_all, pred_train, labels=labels)
df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
sn.heatmap(df_cmx, annot=True, fmt='g' ,square = True, cmap="viridis")
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Estimated Class')
plt.show()

#%% ----------テスト1------------------------
# 機械学習後に検証するテスト入力データの作成
test_condition0 = np.array([[30,31,22,12,15,10,10,15,11,10]]) # ダミーの10計測データを用意

ans = clf.predict(test_condition0) # ダミーデータを投げて出力をしらべる
print(ans)

#%% 学習済み・機械学習モデルの保存
import pickle
with open('model.pickle', mode='wb') as fp:
    pickle.dump(clf, fp)

del clf
#%% 学習済み・機械学習モデルの読み出し
with open('model.pickle', mode='rb') as fp:
    clf = pickle.load(fp)

#%% ----------再テスト1------------------------
# 機械学習後に検証するテスト入力データの作成

ans = clf.predict(test_condition0)
print(ans)


#%% ----------テスト2------------------------
# 機械学習後に検証するテスト入力データの作成 まとめて複数データでテスト
num2=2
test_condition1 = np.random.randint( 0, 10,(num2,10)) 
test_condition2 = np.random.randint(10, 20,(num2,10))  
test_condition3 = np.random.randint(15, 25,(num2,10)) 

test_condition_all = np.vstack([test_condition1, test_condition2, test_condition3])

ans = clf.predict(test_condition_all)
print(ans)


#%% -----------------------------------
