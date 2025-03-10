

import numpy as np
import mne
from mne.io import read_raw_eeglab
from scipy.io import savemat, loadmat
import h5py
import neurokit2 as nk
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import warnings
from scipy.io import savemat
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneGroupOut
# 忽略所有的UserWarning
warnings.filterwarnings('ignore', category=UserWarning)

class Subjective:

    def __init__(self, filename=None):

        m = h5py.File(filename)
        n_trial = m['Subjective']['continous_report'].shape[0]

        self.continous_report = np.zeros((n_trial, 1800))
        self.video_id = np.zeros((n_trial, ))
        self.exp_type = [0] * n_trial
        for i in range(n_trial):
            self.continous_report[i, :] = np.array(m[m['Subjective']['continous_report'][i,0]])
            self.video_id[i] = np.array(m[m['Subjective']['video_id'][i,0]])
            self.exp_type[i] = m[m['Subjective']['type'][i,0]].shape[0]
        self.exp_type = ['WATCH' if x > 4 else 'RATE' for x in self.exp_type]

if __name__ == "__main__":

    print(nk.cite())
    acc = []
    group=[]
    for si in range(0, 24):

        ## 读取连续标签
        s = Subjective(
            "D:/release/200Hz_rawdata/%02d_Subjective.mat" % (si+1),
        )

        ## 读取脑电数据
        raw = mne.io.read_raw_eeglab(
            "D:/release/1000Hz_rawdata_multimodal/%02d.set" % (si+1), 
            eog=(), 
            preload=False, 
            uint16_codec=None, 
            montage_units='auto', 
            verbose=None,
        )
        rawEpoched = mne.Epochs(
            raw,
            tmin=-1,
            tmax=60,
            baseline=(None, 0.0),
        ).load_data()
        print(raw.info, rawEpoched._data.shape)
        
        X = rawEpoched._data[:,-6:, :] # ['PPG', 'RSP', 'EMG-A', 'EDA', 'ECG', 'EMG-B']
        y = rawEpoched.events[:, -1]
        

        for i in range(X.shape[0]):
                
            ppg=X[i, 0, :]
            #ppg_elgendi = nk.ppg_clean(ppg, method='elgendi')
            
            # try:
            #     # 尝试执行可能引发 ValueError 的代码
            #     # nk.ppg_intervalrelated(df, sampling_rate=1000)
            #     df, info = nk.ppg_process(ppg, sampling_rate=1000)
            # except ValueError:
            #     # 如果出现 ValueError，则将 result1 和 result2 赋值为 0
            #     ppg_analyze=pd.DataFrame(np.random.rand(1,ppg_analyze.shape[1]))
                
            # else:
            #     # 如果没有出现 ValueError，则执行其他代码
            #     # ...
            #     df, info = nk.ppg_process(ppg, sampling_rate=1000)
            
            # #df, info = nk.ppg_process(ppg, sampling_rate=1000)
            # if i == 0:
            #     ppg_feature= nk.ppg_analyze(df, sampling_rate=1000)
            # else:
            #     ppg_analyze=nk.ppg_analyze(df, sampling_rate=1000)
            #     ppg_feature = pd.concat([ppg_feature,ppg_analyze], ignore_index=True)
            try:
                # 尝试执行可能引发 ValueError 的代码
                df, info = nk.ppg_process(ppg, sampling_rate=1000)
            except ValueError:
                # 如果出现 ValueError，则将 result1 和 result2 赋值为 0
                ppg_analyze=pd.DataFrame(np.random.rand(1,ppg_analyze.shape[1]))
                
            else:
                # 如果没有出现 ValueError，则执行其他代码
                # ...
                df, info = nk.ppg_process(ppg, sampling_rate=1000)
            
            #df, info = nk.ppg_process(ppg, sampling_rate=1000)
            if i == 0:
                ppg_feature= nk.ppg_intervalrelated(df, sampling_rate=1000)
            else:
                ppg_analyze=nk.ppg_intervalrelated(df, sampling_rate=1000)
                ppg_feature = pd.concat([ppg_feature,ppg_analyze], ignore_index=True)
            
        # 找到 NaN 值的位置
        nan_indices = np.where(ppg_feature.isna()| np.isinf(ppg_feature))

        # 生成随机值
        random_values = np.random.rand(len(nan_indices[0]))

        # 将随机值赋给 NaN 值的位置
        ppg_feature.values[nan_indices] = random_values

        # cols_with_nan = ppg_feature.isna().any()
        # print(cols_with_nan[cols_with_nan].index.tolist())
        # ppg_feature = ppg_feature.drop(cols_with_nan[cols_with_nan].index, axis=1)
        X_ppg_feature = np.array(ppg_feature)
        # w=np.isinf(X_ppg_feature)
        # X_ppg_feature = X_ppg_feature[:, ~w.any(axis=0)]
        X_ppg_feature_scaled = StandardScaler().fit_transform(X_ppg_feature)
        
        if si == 0:
            combined_ppg=X_ppg_feature_scaled
            combined_y = np.array([])
            combined_y = np.append(combined_y ,y,axis=0)
        else :
            combined_ppg=np.append(combined_ppg,X_ppg_feature_scaled, axis=0)
            combined_y = np.append(combined_y ,y,axis=0)
        print(X.shape, y.shape)
        for j in range(X.shape[0]):
            group.append(si)

        X = X_ppg_feature_scaled    
        # 指定留1样本的交叉验证
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        score_list = []
        model = svm.SVC(kernel='linear', C=1.0)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score_list.append((model.score(X_test, y_test)))
            
        #print(np.array(score_list))
        print(np.array(score_list).mean())
        
        with open('D:/icassp_meeting/acc_ppg1.log', 'a') as f:
            f.write(
                "ACC=%.2f\n" % (np.array(score_list).mean()) 
            )
    
    
    # cols_with_nan = combined_ppg.isna().any()
    # print(cols_with_nan[cols_with_nan].index.tolist())
    # combined_ppg = combined_ppg.drop(cols_with_nan[cols_with_nan].index, axis=1)
    # combined_ppg = np.array(combined_ppg)
    # w=np.isinf(combined_ppg)
    # combined_ppg = combined_ppg[:, ~w.any(axis=0)]
    # combined_ppg = StandardScaler().fit_transform(combined_ppg)       
    model = svm.SVC(kernel='linear', C=1.0)
    score_list = []
    logo = LeaveOneGroupOut()
    for train_index, test_index in logo.split(combined_ppg, combined_y, group):
        X_train, X_test = combined_ppg[train_index], combined_ppg[test_index]
        y_train, y_test = combined_y[train_index], combined_y[test_index]
        
        
        model.fit(X_train,y_train)
        preds =  model.predict(X_test)
        score_list.append(( model.score(X_test, y_test)))
    print(np.array(score_list))
    print(np.array(score_list).mean())
    with open('D:/icassp_meeting/acc_ppg2.log', 'w') as f:
        for score in score_list:
            f.write(str(score) + '\n')
        f.write(f'Mean Score: {np.array(score_list).mean()}')
