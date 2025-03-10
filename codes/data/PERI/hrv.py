

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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
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

            ecg = X[i, 4, :]
            # # Clean (filter and detrend)
            # #ecg_processed,info=nk.ecg_process(ecg,simpling_rate=1000)
            # # ecg_cleaned = nk.signal_detrend(ecg)
            # # ecg_cleaned = nk.signal_filter(ecg_cleaned, lowcut=0.5, highcut=10)
            # ecg_cleaned = nk.ecg_clean(ecg,sampling_rate = 1000)
            # peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=1000,correct_artifacts=True)
            # if i == 0:
            #     hrv = nk.hrv(peaks, sampling_rate=1000, show=False)
            # else:
            #     hrv = pd.concat([hrv, nk.hrv(peaks, sampling_rate=1000, show=False)], ignore_index=True)
            
            if si==23 and i==18:
                ecg_fea = pd.concat([ecg_fea,  pd.DataFrame(0, index=A.index, columns=A.columns)], ignore_index=True)
            else:
                df, info = nk.ecg_process(ecg, sampling_rate=1000)
                if i == 0:
                    ecg_fea=nk.ecg_intervalrelated(df, sampling_rate=1000)
                    A=ecg_fea
                else:
                    ecg_fea = pd.concat([ecg_fea, nk.ecg_intervalrelated(df, sampling_rate=1000)], ignore_index=True)
                
                
            
            
        # # 找到 NaN 值的位置
        # nan_indices = np.where(hrv.isna()| np.isinf(hrv))

        # # 生成随机值
        # random_values = np.random.rand(len(nan_indices[0]))

        # # 将随机值赋给 NaN 值的位置
        # hrv.values[nan_indices] = random_values
        
        # for col in hrv.columns:
        #     # 找到 NaN 值的索引
        #     nan_indices = hrv[col].isnull() | np.isinf(hrv[col])#hrv[col].isnull()
        #     # 生成随机值并替换 NaN 值
        #     hrv.loc[nan_indices, col] = np.random.rand()
        


        # # cols_with_nan = hrv.isna().any()
        # # print(cols_with_nan[cols_with_nan].index.tolist())
        # # hrv = hrv.drop(cols_with_nan[cols_with_nan].index, axis=1)
        # X_hrv = np.array(hrv)
        # # w=np.isinf(X_HRV)
        # # X_HRV = X_HRV[:, ~w.any(axis=0)]
        
        
        ecg_fea=ecg_fea.astype(float)
        nan_indices = np.where(ecg_fea.isna()| np.isinf(ecg_fea))
        # 生成随机值
        random_values = np.random.rand(len(nan_indices[0]))
        # 将随机值赋给 NaN 值的位置
        ecg_fea.values[nan_indices] = random_values
        X_ECG = np.array(ecg_fea)
        
        
        
        X_HRV_scaled = StandardScaler().fit_transform(X_ECG)

        
        
        
        if si == 0:
            combined_HRV=X_HRV_scaled
            combined_y = np.array([])
            combined_y = np.append(combined_y ,y,axis=0)
        else :
            min_len = min(combined_HRV.shape[1], X_HRV_scaled.shape[1])
            
            combined_HRV=np.append(combined_HRV[:,:min_len],X_HRV_scaled[:,:min_len],axis=0)
            combined_y = np.append(combined_y ,y,axis=0)
        print(X.shape, y.shape)
        for j in range(X.shape[0]):
            group.append(si)

        X = X_HRV_scaled
        
        # 指定留1样本的交叉验证
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        
        # 用于比较模型的分类器，这里以RandomForestClassifier为例
        model = svm.SVC(kernel='linear', C=1.0)
        
        score_list = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score_list.append((model.score(X_test, y_test)))
            
        #print(np.array(score_list))
        print(np.array(score_list).mean())
        

        with open('D:/icassp_meeting/acc_hrv1.log', 'a') as f:
            f.write(
                "ACC=%.2f\n" % (np.array(score_list).mean()) 
            )
        
    
    
    # cols_with_nan = combined_HRV.isna().any()
    # print(cols_with_nan[cols_with_nan].index.tolist())
    # combined_HRV = combined_HRV.drop(cols_with_nan[cols_with_nan].index, axis=1)
    # combined_HRV = np.array(combined_HRV)
    # w=np.isinf(combined_HRV)
    # combined_HRV = combined_HRV[:, ~w.any(axis=0)]
    
    
    
    # X_train, X_test, y_train, y_test = train_test_split(combined_HRV, combined_y, test_size=0.2, random_state=42)

    # # 3. 定义 SVM 模型
    # svm = SVC()

    # # 4. 定义超参数搜索空间
    # param_grid = {
    #     'C': [0.001, 0.01, 0.1, 0.2 , 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1],  # 正则化参数
    #     'kernel': ['linear'],  # 核函数
    # }

    # # 5. 创建 GridSearchCV 对象
    # grid_search = GridSearchCV(
    #     estimator=svm,
    #     param_grid=param_grid,
    #     scoring='accuracy',  # 使用准确率作为评价指标
    #     cv=5,  # 使用 5 折交叉验证
    #     verbose=3,
    #     n_jobs=3,
    # )

    # # 6. 训练模型
    # grid_search.fit(X_train, y_train)

    # # 7. 获取最佳模型和参数
    # best_model = grid_search.best_estimator_
    # best_params = grid_search.best_params_

    # # 8. 在测试集上评估模型
    # y_pred = best_model.predict(X_test)
    # accuracy = best_model.score(X_test, y_test)

    # # 9. 打印结果
    # print(f"最佳参数：{best_params}")
    # print(f"测试集准确率：{accuracy}")
    
    
    
    
        
    model = svm.SVC(kernel='linear', C=1.0)
    score_list = []
    logo = LeaveOneGroupOut()
    for train_index, test_index in logo.split(combined_HRV, combined_y, group):
        X_train, X_test = combined_HRV[train_index], combined_HRV[test_index]
        y_train, y_test = combined_y[train_index], combined_y[test_index]
        
        
        model.fit(X_train,y_train)
        preds =  model.predict(X_test)
        score_list.append(( model.score(X_test, y_test)))
    print(np.array(score_list))
    print(np.array(score_list).mean())
    with open('D:/icassp_meeting/acc_hrv2.log', 'w') as f:
        for score in score_list:
            f.write(str(score) + '\n')
        f.write(f'Mean Score: {np.array(score_list).mean()}')

    
