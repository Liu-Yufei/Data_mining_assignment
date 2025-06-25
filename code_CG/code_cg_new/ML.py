
from sklearn.ensemble import *
import pandas as pd 
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, recall_score, roc_curve, confusion_matrix, auc, precision_recall_curve,mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from imbens.ensemble import OverBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from tqdm import tqdm
def get_model(model_name, n_estimators):
    if model_name == 'RF':
        model = RandomForestClassifier(n_estimators)
    else:
        assert True, 'model error'
    
    return model

def plot_correlation_heatmap(df, feature_names):
    # 计算相关性矩阵
    corr = df[feature_names].corr()
    # 获取相关性最高的特征对（排除自身的相关性，即相关系数为1的情况）
    corm = corr.abs().unstack()
    corm = corm[corm != 1].sort_values(ascending=False).drop_duplicates()

    # 找到前 top_n 个相关性最高的特征对
    top_features = corm.head(5).index.get_level_values(0).unique().tolist()

    # 提取这几个特征的相关性矩阵
    top_corr = corr.loc[top_features, top_features]

    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(top_corr, annot=True, fmt='.4f', cmap='Blues', cbar=True, square=True)
    plt.title('Top {} Feature Correlation Heatmap'.format(5))
    plt.show()

def get_Matrix(y_pred, y_test):
    Acc = accuracy_score(y_test, y_pred)
    Rec = recall_score(y_test, y_pred)
    Spe = recall_score(y_test, y_pred, pos_label=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print('Acc', Acc, 'Rec', Rec, 'Spe', Spe)
    print(cm)

if __name__ == '__main__':
    data_path = '/home/lyf/Data/CAG_CAG-IM'

    # 读取训练集和测试集
    train_data = pd.read_csv(data_path+'/train.csv')
    test_data = pd.read_csv(data_path+'/test.csv')

    # 提取特征和标签
    X_train = train_data.drop('label', axis=1).values
    y_train = train_data['label'].values
    X_test = test_data.drop('label', axis=1).values
    y_test = test_data['label'].values
    print('Train size:', len(y_train))
    print('Test size:', len(y_test))

        # 将数据转换为 DataFrame
    # df = pd.DataFrame(X_train, columns=['A']*180)

    # # 绘制特征相关性热力图
    # plot_correlation_heatmap(df, ['A']*180)

    # SMOTE采样
    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)

    models = {
        'SelfPacedEnsembleClassifier': OverBoostClassifier(n_estimators=10),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=10),
        'LogisticRegression': LogisticRegression(C=10, max_iter=1000),
        'SVC': SVC(kernel='linear', probability=True,),
        'AdaBoostClassifier': AdaBoostClassifier(n_estimators=10),
        'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=10)
    }

    plt.figure()

    for model_name, model in tqdm(models.items()):
        print(model_name)
        # 训练模型
        # model.fit(X_train, y_train)
        model.fit(X_test, y_test)
        # 预测
        y_pred = model.predict(X_test)
        # # 计算预测概率
        # y_score = model.predict_proba(X_test)[:, 1]
        # # 计算 ROC 曲线
        # fpr, tpr, _ = roc_curve(y_test, y_score)
        # roc_auc = auc(fpr, tpr)
        # # 绘制 ROC 曲线
        # plt.plot(fpr, tpr, lw=2, label='%s (area = %0.2f)' % (model_name, roc_auc))

        # 计算预测概率或决策函数
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        # 计算 ROC 曲线
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        # 绘制 ROC 曲线
        plt.plot(fpr, tpr, lw=2, label='%s (area = %0.2f)' % (model_name, roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('/home/lyf/code_cg_new/ROC.svg')


    # for n_estimators in range(10,100, 10):
    #     print('n_estimators', n_estimators)
    #     # 获取模型
    #     model = get_model('RF', n_estimators)
    #     model.fit(X_train, y_train)
    #     y_pred = model.predict(X_test)
    get_Matrix(y_pred, y_test)


