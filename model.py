import time
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score ,roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report


def GMM(name, model, x_train, x_val, y_val,start,std):
    model = model(n_components=2)
    clf = model.fit(std.transform(x_train))
    y_val_cat = clf.predict(std.transform(x_val))

    confusion = confusion_matrix(y_true=y_val, y_pred=y_val_cat)
    report = classification_report(y_true=y_val, y_pred=y_val_cat)
    accuracy = accuracy_score(y_val,y_val_cat)
    end = time.time()

    print (f'模型{name}, 耗时 {(end-start)/60} minutes:')
    print (f'使用clf.score计算, 测试集的准确度为{accuracy}')
    print (f'测试集的混淆矩阵为:\n{confusion}')
    print (f'测试集的分类报告为:\n{report}')

def model_train(name, model, x_train, y_train, x_val, y_val):
    if name in ['logistic', 'bayes', 'svm', 'mixture', 'KNN']:
        start = time.time()
        std = StandardScaler()
        std = std.fit(x_train)
        if name == 'svm':
            model = model(max_iter=500,probability=True,kernel='poly')
        elif name == 'mixture':
            # 聚类算法
            GMM(name, model, x_train, x_val, y_val,start,std)
            return 
        elif name == 'KNN':
            model = model(n_jobs=-1)
        else:
            model = model()
        clf = model.fit(std.transform(x_train),y_train)
        y_val_cat = clf.predict(std.transform(x_val))
        y_val_pred = clf.predict_proba(std.transform(x_val))

        confusion = confusion_matrix(y_true=y_val, y_pred=y_val_cat)
        report = classification_report(y_true=y_val, y_pred=y_val_cat)
        accuracy = clf.score(std.transform(x_val),y_val)
        roc = roc_auc_score(y_true=y_val, y_score=y_val_pred[:,1])
        end = time.time()
        
        print (f'模型{name}, 耗时 {(end-start)/60} minutes:')
        print (f'使用clf.score计算, 测试集的准确度为{accuracy}')
        print (f'测试集的ROC面积为{roc}')
        print (f'测试集的混淆矩阵为:\n{confusion}')
        print (f'测试集的分类报告为:\n{report}')
        return model
    elif name in ['decision_tree', 'GBDT', 'XGBoost']:
        start = time.time()
        if name in ['XGBoost']:
            # model = model(n_jobs=-1)
            model = model(
                n_jobs=-1,
                max_depth=8,
                n_estimators=100,
                min_child_weight=300, 
                colsample_bytree=0.8, 
                subsample=0.8, 
                eta=0.3,    
                seed=42)
        else:
            model = model()
        clf = model.fit(x_train,y_train)
        y_val_cat = clf.predict(x_val)
        y_val_pred = clf.predict_proba(x_val)
        
        confusion = confusion_matrix(y_true=y_val, y_pred=y_val_cat)
        report = classification_report(y_true=y_val, y_pred=y_val_cat)
        accuracy = clf.score(x_val,y_val)
        roc = roc_auc_score(y_true=y_val, y_score=y_val_pred[:,1])
        end = time.time()

        print (f'模型{name}:')
        print (f'模型{name}, 耗时 {(end-start)/60} minutes:')
        print (f'使用clf.score计算, 测试集的准确度为{accuracy}')
        print (f'测试集的ROC面积为{roc}')
        print (f'测试集的混淆矩阵为:\n{confusion}')
        print (f'测试集的分类报告为:\n{report}')
        return model
    else:
        print ('找不到此模型')