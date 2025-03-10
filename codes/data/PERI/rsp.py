

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
                
            rsp=X[i, 1, :]
            # #ppg_elgendi = nk.ppg_clean(ppg, method='elgendi')
            
            # # Clean signal
            # cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
            # # Extract peaks
            # df, peaks_dict = nk.rsp_peaks(cleaned) 
            # info = nk.rsp_fixpeaks(peaks_dict)
            # formatted = nk.signal_formatpeaks(info, desired_length=len(cleaned),peak_indices=info["RSP_Peaks"])
            # # Extract rate
            # rsp_rate = nk.rsp_rate(cleaned, peaks_dict, sampling_rate=1000)
            # # if i == 0:
            # #     rrv = nk.rsp_rrv(rsp_rate, info, sampling_rate=1000, show=True)
            # # else:
            # #     rrv = pd.concat([rrv,nk.rsp_rrv(rsp_rate, info, sampling_rate=1000, show=True)], ignore_index=True)
            
            
            
            
            # try:
            #     # 尝试执行可能引发 ValueError 的代码
            #     nk.rsp_rrv(rsp_rate, info, sampling_rate=1000, show=False)
            # except ValueError:
            #     # 如果出现 ValueError，则
            #     if i == 0:
            #         random=pd.DataFrame(np.random.rand(1,rrv.shape[1]))
            #         random.columns = [rrv.columns[j] for j in range(len(random.columns))]
            #         rrv = random
            #     else:
            #         random=pd.DataFrame(np.random.rand(1,rrv.shape[1]))
            #         random.columns = [rrv.columns[j] for j in range(len(random.columns))]
            #         rrv = pd.concat([rrv,random],ignore_index=True)
                
                
            # else:
            #     # 如果没有出现 ValueError，则执行其他代码
            #     # ...
            #     if i == 0:
            #         rrv = nk.rsp_rrv(rsp_rate, info, sampling_rate=1000, show=False)
            #     else:
            #         rrv = pd.concat([rrv,nk.rsp_rrv(rsp_rate, info, sampling_rate=1000, show=False)], join='inner',ignore_index=True)
            #ppg_elgendi = nk.ppg_clean(ppg, method='elgendi')
            df, info = nk.rsp_process(rsp, sampling_rate=1000)
            # # Clean signal
            # cleaned = nk.rsp_clean(rsp, sampling_rate=1000)
            # # Extract peaks
            # df, peaks_dict = nk.rsp_peaks(cleaned) 
            # info = nk.rsp_fixpeaks(peaks_dict)
            # formatted = nk.signal_formatpeaks(info, desired_length=len(cleaned),peak_indices=info["RSP_Peaks"])
            # # Extract rate
            # rsp_rate = nk.rsp_rate(cleaned, peaks_dict, sampling_rate=1000)
            try:
                # 尝试执行可能引发 ValueError 的代码
                nk.rsp_intervalrelated(df, sampling_rate=1000)
            except ValueError:
                # 如果出现 ValueError，则
                if i == 0:
                    random=pd.DataFrame(np.random.rand(1,rsp_fea.shape[1]))
                    random.columns = [rsp_fea.columns[j] for j in range(len(random.columns))]
                    rsp_fea = random
                else:
                    random=pd.DataFrame(np.random.rand(1,rsp_fea.shape[1]))
                    random.columns = [rsp_fea.columns[j] for j in range(len(random.columns))]
                    rsp_fea = pd.concat([rsp_fea,random],ignore_index=True)  
            else:
                # 如果没有出现 ValueError，则执行其他代码
                # ...
                if i == 0:
                    rsp_fea = nk.rsp_intervalrelated(df, sampling_rate=1000)
                else:
                    rsp_fea = pd.concat([rsp_fea,nk.rsp_intervalrelated(df, sampling_rate=1000)], join='inner',ignore_index=True)   

            
        # 找到 NaN 值的位置
        nan_indices = np.where(rsp_fea.isna()| np.isinf(rsp_fea))

        # 生成随机值
        random_values = np.random.rand(len(nan_indices[0]))

        # 将随机值赋给 NaN 值的位置
        rsp_fea.values[nan_indices] = random_values
        
        
        
        # cols_with_nan = rrv.isna().any()
        # if cols_with_nan.all():
        # # 如果全是 True，则创建一个随机值数组
        #     Xrrv=np.random.rand((rrv.shape[0],rrv.shape[1]))
        # else:
        #     print(cols_with_nan[cols_with_nan].index.tolist())
        #     rrv= rrv.drop(cols_with_nan[cols_with_nan].index, axis=1)
        #     X_rrv = np.array(rrv)
        #     w=np.isinf(X_rrv)
        #     X_rrv = X_rrv[:, ~w.any(axis=0)]
        X_rrv_scaled=np.array(rsp_fea)
        X_rrv_scaled = StandardScaler().fit_transform(X_rrv_scaled)
        
        
        # try:
        #         # 尝试执行可能引发 ValueError 的代码
        #         StandardScaler().fit_transform(X_rrv)
        # except ValueError:
        #     # 如果出现 ValueError，则将 result1 和 result2 赋值为 0
        #     rrv = pd.concat([rrv,pd.DataFrame(np.zeros((1,rrv.shape[1])))], ignore_index=True)
            
        # else:
        #     # 如果没有出现 ValueError，则执行其他代码
        #     # ...
        #     if i == 0:
        #         rrv = nk.rsp_rrv(rsp_rate, info, sampling_rate=1000, show=True)
        #     else:
        #         rrv = pd.concat([rrv,nk.rsp_rrv(rsp_rate, info, sampling_rate=1000, show=False)], ignore_index=True)
            
        
        
        if si == 0:
            combined_rrv=X_rrv_scaled
            combined_y = np.array([])
            combined_y = np.append(combined_y ,y,axis=0)
        else :
            combined_rrv=np.append(combined_rrv,X_rrv_scaled, axis=0)
            combined_y = np.append(combined_y ,y,axis=0)
        print(X.shape, y.shape)
        for j in range(X.shape[0]):
            group.append(si)


        # 指定留1样本的交叉验证
        X = X_rrv_scaled  
        kf = KFold(n_splits=10, shuffle=True, random_state=42)  
        score_list = []
        model = svm.SVC(kernel='linear', C=1.0)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score_list.append((model.score(X_test, y_test)))
        print(np.array(score_list).mean())
        with open('D:/icassp_meeting/acc_rrv1.log', 'a') as f:
            f.write(
                "ACC=%.2f\n" % (np.array(score_list).mean()) 
            )
    # combined_rrv.fillna(0, inplace=True)
    # cols_with_nan = combined_rrv.isna().any()
    # print(cols_with_nan[cols_with_nan].index.tolist())
    # # combined_rrv = combined_rrv.drop(cols_with_nan[cols_with_nan].index, axis=1)
    # # combined_rrv = np.array(combined_rrv)
    # z = combined_rrv.drop(cols_with_nan[cols_with_nan].index, axis=1)
    # z = np.array(z)
    # # w=np.isinf(combined_rrv)
    # # combined_rrv = combined_rrv[:, ~w.any(axis=0)]
    # w=np.isinf(z)
    # z = z[:, ~w.any(axis=0)]
    # #combined_rrv = StandardScaler().fit_transform(combined_rrv)       
    model = svm.SVC(kernel='linear', C=1.0)
    score_list = []
    logo = LeaveOneGroupOut()
    for train_index, test_index in logo.split(combined_rrv, combined_y, group):
        X_train, X_test = combined_rrv[train_index,:], combined_rrv[test_index]
        y_train, y_test = combined_y[train_index], combined_y[test_index]
        
        
        model.fit(X_train,y_train)
        preds =  model.predict(X_test)
        score_list.append(( model.score(X_test, y_test)))
    print(np.array(score_list))
    print(np.array(score_list).mean())
    with open('D:/icassp_meeting/acc_rrv2.log', 'w') as f:
        for score in score_list:
            f.write(str(score) + '\n')
        f.write(f'Mean Score: {np.array(score_list).mean()}')
