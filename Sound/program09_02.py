#%% 機械学習（信号解析で信号特徴が判明した後の自動判別化） 解析した生データを直接渡すのではなく、周波数解析をはさんでやる
#周波数解析の意味->複雑な波の情報を振幅と周波数で表現するように簡略化して解析する
#リセット用ライブラリ
from IPython import get_ipython 
#リセット
get_ipython().magic('reset -sf')

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

#%% 1種類のデータセット数
num = 40

A = np.zeros([1000, 1 + 4*num]) #配列の作成

#時間[秒]
A[:,0] = np.arange(0,1.000,0.001)

# 4種類の振幅と周波数の異なる信号(ノイズ添加：位相，振幅)
A[:, 1+0*num:1+1*num] = 1*np.sin(2*np.pi*(np.array([A[:,0]]).T)*2+2*np.pi*np.random.rand(num)) + 1*(2*np.random.rand(len(A[:,0]),num) - 1) #信号1[V]
A[:, 1+1*num:1+2*num] = 2*np.sin(2*np.pi*(np.array([A[:,0]]).T)*4+2*np.pi*np.random.rand(num)) + 1*(2*np.random.rand(len(A[:,0]),num) - 1) #信号2[V]
A[:, 1+2*num:1+3*num] = 3*np.sin(2*np.pi*(np.array([A[:,0]]).T)*6+2*np.pi*np.random.rand(num)) + 1*(2*np.random.rand(len(A[:,0]),num) - 1) #信号3[V]
A[:, 1+3*num:1+4*num] = 4*np.sin(2*np.pi*(np.array([A[:,0]]).T)*8+2*np.pi*np.random.rand(num)) + 1*(2*np.random.rand(len(A[:,0]),num) - 1) #信号4[V]

plt.close("all")
plt.figure(1) # 生データのプロット
plt.rcParams['figure.figsize'] = (9.0, 9.0) #グラフのサイズ設定
plt.rcParams['font.size'] = 11     #全体のフォントサイズ

plt.subplot(2,2,1)
plt.plot(A[:,0],A[:,  1: 41]);
plt.ylim(-4,+4)
plt.grid()

plt.subplot(2,2,2)
plt.plot(A[:,0],A[:, 41: 81]);
plt.ylim(-4,+4)
plt.grid()

plt.subplot(2,2,3)
plt.plot(A[:,0],A[:, 81:121]);
plt.ylim(-4,+4)
plt.grid()

plt.subplot(2,2,4)
plt.plot(A[:,0],A[:,121:161]);
plt.ylim(-4,+4)
plt.grid()


#%% 周波数解析（＝特徴抽出）

# グラフを表示
plt.figure(2) #周波数解析後のデータプロット
plt.clf()
plt.rcParams['figure.figsize'] = (9.0, 9.0) #グラフのサイズ設定
plt.rcParams['font.size'] = 11     #全体のフォントサイズ

for i in range(0,4*num):

    F = 1/(A[1,0]-A[0,0]) #サンプリング周波数[Hz]
    L = len(A[:,0]) #信号のデータ数 
    y = A[:,i+1] #周波数解析対象の信号
    
    #FFT解析部
    temp1 = np.log2(L)
    temp2 = np.ceil(temp1)
    NFFT = 2**int(np.log2(2**temp2)) #Lのデータ数以上の最小の 2 のべき乗の指数
    Y = np.fft.fft(y, NFFT)/L
    Freq = F/2*np.linspace(0, 1, int(NFFT/2)+1) #周波数（横軸）
    Amp = 2*np.abs(Y[:int(NFFT/2)+1]) #振幅（縦軸）

    if i == 0:
        B = np.zeros([len(Freq), 1 + 4*num]) #配列の作成
        B[:,0] = Freq

    B[:,i+1] = Amp

    plt.subplot(2,1,1)
    plt.plot(A[:,0], A[:,i+1])
        
    plt.subplot(2,1,2)
    plt.plot(Freq, Amp)

# 生データを表示
plt.subplot(2,1,1)
plt.grid()
plt.xlim(0,1)
plt.ylim(-5,5)
plt.xlabel('Time[s]')
plt.ylabel('Voltage[V]')
plt.title('Raw signal')

# FFT解析結果を表示
plt.subplot(2,1,2)
plt.grid()
plt.xlim(0,20)
plt.ylim(0,4)
plt.xlabel('Frequency[Hz]')
plt.ylabel('Amplitude[V]')
plt.title('FFT result')

plt.tight_layout()
plt.show()

#%% 学習データの生成

# 入力部分：周波数0.5-12Hzまでのデータを抽出 ピークが出ている所に絞って解析させる
### 以下を変更する
freqStart=0.5 #バンド幅の設定
freqEnd=12 # バンド幅の設定
###

startIndex=np.where(B[:,0]>=freqStart)[0][0] # 周波数範囲の確認（開始位置）
endIndex=np.where(B[:,0]<=freqEnd)[0][-1]  # 周波数範囲の確認（終了位置）
C = B[startIndex:endIndex, 1:]     # 該当周波数範囲の振幅値のデータ抽出

print(B[startIndex:endIndex, 0])

learning_data_all = C.T


#%% 理想的な出力部分：1種類目「1」，2種類目「2」，3種類目「3」，4種類目「4」
target1 = np.full(num, 1)
target2 = np.full(num, 2)
target3 = np.full(num, 3)
target4 = np.full(num, 4)

# 上記のデータを統合して学習データとする
target_all = np.hstack([target1, target2, target3, target4])

plt.figure(3)
plt.clf()
plt.rcParams['figure.figsize'] = (9.0, 9.0) #グラフのサイズ設定
plt.rcParams['font.size'] = 11     #全体のフォントサイズ

plt.subplot(2,2,1)
plt.grid()
plt.ylim(0,4)
plt.xlabel('Frequency[Hz]')
plt.ylabel('Amplitude[V]')
plt.title('FFT result')

plt.subplot(2,2,2)
plt.grid()
plt.ylim(0,4)
plt.xlabel('Frequency[Hz]')
plt.ylabel('Amplitude[V]')
plt.title('FFT result')

plt.subplot(2,2,3)
plt.grid()
plt.ylim(0,4)
plt.xlabel('Frequency[Hz]')
plt.ylabel('Amplitude[V]')
plt.title('FFT result')

plt.subplot(2,2,4)
plt.grid()
plt.ylim(0,4)
plt.xlabel('Frequency[Hz]')
plt.ylabel('Amplitude[V]')
plt.title('FFT result')

for i in range(0,len(target_all)):

    if target_all[i]==1:
        plt.subplot(2,2,1)        
        plt.plot(learning_data_all[i,:].T)

    elif target_all[i]==2:
        plt.subplot(2,2,2)
        plt.plot(learning_data_all[i,:].T)

    elif target_all[i]==3:
        plt.subplot(2,2,3)
        plt.plot(learning_data_all[i,:].T)

    elif target_all[i]==4:
        plt.subplot(2,2,4)
        plt.plot(learning_data_all[i,:].T)

plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
#plt.tight_layout()
plt.show()

#%%
#-------------------------------------
# 学習用データのファイル保存
#-------------------------------------
learning_data_set=np.hstack([learning_data_all,np.array([target_all]).T])

# ファイル保存
df = pd.DataFrame(learning_data_set)
df.to_csv('learning_data_set2.csv', index=False, header=False)


#%% SVMを用いて分類 基本いじらない

# 非線形SVMで2次元の多項式カーネル
# poly: 多項式カーネル
# degree: 多項式カーネル関数の次数
clf = svm.SVC(kernel='poly', degree=2)
clf.fit(learning_data_all, target_all)


#%% 学習済み・機械学習モデルの保存
import pickle

with open('model1.pickle', mode='wb') as fp:
    pickle.dump(clf, fp)


#%% トレーニングデータに対する精度
pred_train = clf.predict(learning_data_all)
accuracy_train = accuracy_score(target_all, pred_train)
print('トレーニングデータに対する正解率： %.3f' % accuracy_train)

#------------------グラフ--------------------
# グラフ4
plt.figure(4)
plt.clf()
plt.rcParams['figure.figsize'] = (9.0, 9.0) #グラフのサイズ設定
plt.rcParams['font.size'] = 11     #全体のフォントサイズ

X0, X1 = learning_data_all[:, 0], learning_data_all[:, 1]
plt.scatter(X0, X1, c=target_all, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
plt.xlabel('column_1')
plt.ylabel('column_2')
plt.title('SVC with polynomial (degree 2) kernel')

plt.show()


# グラフ5
plt.figure(5)
plt.clf()
plt.rcParams['figure.figsize'] = (9.0, 9.0) #グラフのサイズ設定
plt.rcParams['font.size'] = 11     #全体のフォントサイズ

labels = sorted(list(set(target_all.astype('int64'))))
cmx_data = confusion_matrix(target_all, pred_train, labels=labels)
df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
sn.heatmap(df_cmx, annot=True, fmt='g' ,square = True, cmap="viridis")
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Estimated Class')

plt.show()

#%%---テスト---
# 機械学習後に検証するテスト入力データの作成

D = np.zeros([1000, 5]) 
D[:,0] = np.arange(0,1.00,0.001)
D[:,1] = 1*np.sin(2*np.pi*D[:,0]*2) #信号1[V]
D[:,2] = 2*np.sin(2*np.pi*D[:,0]*4) #信号2[V]
D[:,3] = 3*np.sin(2*np.pi*D[:,0]*6) #信号3[V]
D[:,4] = 4*np.sin(2*np.pi*D[:,0]*8) #信号4[V]

for i in range(4):
    F = 1/(D[1,0]-D[0,0]) #サンプリング周波数[Hz]
    L = len(D[:,0]) #信号のデータ数 
    y = D[:,i+1] #周波数解析対象の信号
    
    #FFT解析部
    temp1 = np.log2(L)
    temp2 = np.ceil(temp1)
    NFFT = 2**int(np.log2(2**temp2)) #Lのデータ数以上の最小の 2 のべき乗の指数
    Y = np.fft.fft(y, NFFT)/L
    Freq = F/2*np.linspace(0, 1, int(NFFT/2)+1) #周波数（横軸）
    Amp = 2*np.abs(Y[:int(NFFT/2)+1]) #振幅（縦軸）
    if i == 0:
        E = np.zeros([len(Freq), 5]) #配列の作成
        E[:,0] = Freq
    E[:,i+1] = Amp

F = E[startIndex:endIndex, 1:]

# 予測結果
ans = clf.predict(F[:,0:].T)
print(ans)