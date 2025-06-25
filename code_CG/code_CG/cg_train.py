# import ReadData_Tool
from sklearn.model_selection import train_test_split
# from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_curve, confusion_matrix, auc, precision_recall_curve,mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import random
# import pandas as pd
from data_get import *
from imblearn.over_sampling import SMOTE
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from options import *
from imbens.ensemble import *
import csv

def custom_roc_curve(y_true, y_scores, num_thresholds=50):
    # Generate custom thresholds
    thresholds = np.linspace(max(y_scores), min(y_scores), num_thresholds)
    
    # Initialize lists to hold TPR and FPR values
    tpr_list = [0.0]  # Start with (0, 0)
    fpr_list = [0.0]  # Start with (0, 0)
    
    # Calculate TPR and FPR for each threshold
    for threshold in thresholds:
        # Binarize the scores based on the current threshold
        y_pred = (y_scores >= threshold).astype(int)
        
        # Calculate TPR and FPR
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    tpr_list.append(1.0)  # End with (1, 1)
    fpr_list.append(1.0)  # End with (1, 1)
    
    return fpr_list, tpr_list, thresholds


def get_model(args,n_estimators):
    if args.model == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        pipeline = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    elif args.model == 'AB':
        from sklearn.ensemble import AdaBoostClassifier
        pipeline = AdaBoostClassifier(n_estimators=n_estimators, random_state=42)
    elif args.model == 'DT':
        from sklearn.tree import DecisionTreeClassifier
        pipeline = DecisionTreeClassifier(random_state=42)
    elif args.model == 'SVM':
        from sklearn.svm import SVC
        pipeline = SVC(random_state=42,probability=True)
    elif args.model == 'LR':
        from sklearn.linear_model import LogisticRegression
        pipeline = LogisticRegression(random_state=42,probability=True)
    elif args.model == 'BC':
        from sklearn.ensemble import BaggingClassifier
        from sklearn.ensemble import RandomForestClassifier
        pipeline = BaggingClassifier(base_estimator=RandomForestClassifier(), n_estimators=n_estimators, random_state=1034)
    elif args.model == 'RF+':
        from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        param_dist = {
            'n_estimators': [int(x) for x in np.linspace(start=10, stop=300, num=10)],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [int(x) for x in np.linspace(40, 110, num=11)],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf':[1, 2, 4, 8, 16], 
            'bootstrap': [True, False]
        }
        param_grid = {
            'n_estimators': [int(x) for x in np.linspace(start=10, stop=100, num=10)],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 8, 16], 
            'bootstrap': [True, False]
        }
        search  = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2)
        search.fit(X_train, y_train)
        pipeline = RandomForestClassifier(**search.best_params_)
        print(f"Best parameters found by random search: {search.best_params_}")
    else:
        pipeline = None
    return pipeline
def random2list(l1,l2):
    cob = list(zip(l1,l2))
    random.shuffle(cob)
    l1[:], l2[:] = zip(*cob)
    return l1, l2

def our_train_test_split(features, Y, test_size=0.3):
    # 按照每个类别数量的比例划分训练集和测试集
    cls_num = {}
    for i,y in enumerate(Y):
        if y not in cls_num:
            cls_num[y] = [features[i]]
        else:
            cls_num[y].append(features[i])
    
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for y in cls_num.keys():
        random.shuffle(cls_num[y])
        split_index = int(len(cls_num[y])*test_size)
        list1 = cls_num[y][:split_index]
        list2 = cls_num[y][split_index:]
        test_x.extend(list1)
        train_x.extend(list2)
        test_y.extend([y]*len(list1))
        train_y.extend([y]*len(list2))
    print(len(test_y))
    train_x, train_y = random2list(train_x, train_y)
    test_x, test_y = random2list(test_x, test_y)
    print(len(test_y))
    return train_x, test_x, train_y, test_y
    

if __name__ == '__main__':
    # 读入数据
    args = parse_args()
    random.seed(args.seed)
    if args.cls == 2:
        # cg_path = "/home/lyf/code_cg/data_5cls/CAG"
        # health_path = "/home/lyf/code_cg/data_5cls/CAG-IM"
        # Y,features = get_feature_2(cg_path,health_path)
        test_csv = "/home/lyf/Data/CAG_CAG-IM/Conv_LSTM_hjz/test.csv"
        train_csv = "/home/lyf/Data/CAG_CAG-IM/Conv_LSTM_hjz/train.csv"
        # test_csv = "/home/lyf/Data/CAG_CNAG/Conv_HP/test.csv"
        # train_csv = "/home/lyf/Data/CAG_CNAG/Conv_HP/train.csv"
        # X_train, X_test, y_train, y_test=get_feature_HP(train_csv,test_csv)
        X_train, X_test, y_train, y_test=get_feature(train_csv,test_csv)
    elif args.cls == 5:
        cag_path = "/home/lyf/code_cg/data_5cls/CAG"
        cag_im_path = "/home/lyf/code_cg/data_5cls/CAG-IM"
        cag_cancer_path = "/home/lyf/code_cg/data_5cls/CAG-cancer"
        jk_path = "/home/lyf/code_cg/data_5cls/JK"
        cnag_path = "/home/lyf/code_cg/data_5cls/CNAG"
        Y,features = get_feature_5(cag_path,cag_im_path,cag_cancer_path,jk_path,cnag_path)
    # Y, features = filter_feature(Y, features)
    # assert False
    mse_loss = []
    # X_train, X_test, y_train, y_test = train_test_split(features, Y, test_size=0.3, random_state=args.seed)
    # X_train, X_test, y_train, y_test = our_train_test_split(features, Y, test_size=0.3)
    print('Train size:', len(y_train))
    print('Test size:', len(y_test))

    Accuracys = []
    Recalls = []
    Specificitys = []
    AUCs = []
    APs = []
    best_acc = 0
    # 使用SMOTE进行上采样
    # for random_state in range(10,50,5):
    #     args.seed = random_state
    if True:
        args.seed =  45
        if args.cls == 2:
            sm = SMOTE(random_state=args.seed)
        else:
            sm = SMOTE(random_state=args.seed, k_neighbors=2)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

        # for n_estimators in range(10,50,5):
        if True:
            # n_estimators = 30 # RF
            # n_estimators = 60 # AB
            n_estimators = 45
            print('ransom_state:',args.seed)
            print('n_estimators:',n_estimators)
            pipeline = get_model(args,n_estimators)

            if pipeline is None:
                print('Invalid model name:', args.model)
                exit(1)
            pipeline.fit(X_train, y_train)
            # pipeline.fit(X_train_res, y_train_res)
            # best_model.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            # imbalance 训练模型 
            # OverBoostClassifier #
            '''
            Accuracy: 0.528169014084507
            Recall: [0.47058824 0.4893617  0.45454545 0.64444444]
            Specificity: [0.47058824 0.4893617  0.45454545 0.64444444]
            Confusion matrix:
            [[ 8  6  2  1]
            [ 5 23  7 12]
            [ 2  7 15  9]
            [ 3  7  6 29]]
            '''
            # SMOTEBoostClassifier
            '''
            Accuracy: 0.5070422535211268
            Recall: [0.52941176 0.38297872 0.60606061 0.55555556]
            Specificity: [0.52941176 0.38297872 0.60606061 0.55555556]
            Confusion matrix:
            [[ 9  6  0  2]
            [ 5 18 13 11]
            [ 1  6 20  6]
            [ 3  8  9 25]]
            '''
            # KmeansSMOTEBoostClassifier
            '''
            n_estimators: 30
            Accuracy: 0.5
            Recall: [0.47058824 0.53191489 0.42424242 0.53333333]
            Specificity: [0.47058824 0.53191489 0.42424242 0.53333333]
            Confusion matrix:
            [[ 8  9  0  0]
            [ 6 25  7  9]
            [ 1 10 14  8]
            [ 2 12  7 24]]
            '''
            # SMOTEBaggingClassifier
            '''
            n_estimators: 90
            Accuracy: 0.5563380281690141
            Recall: [0.58823529 0.59574468 0.48484848 0.55555556]
            Specificity: [0.58823529 0.59574468 0.48484848 0.55555556]
            Confusion matrix:
            [[10  5  1  1]
            [ 1 28 12  6]
            [ 2  4 16 11]
            [ 2  9  9 25]]
            '''
            # # OverBaggingClassifier
            # pipeline = OverBaggingClassifier(random_state=args.seed, n_estimators=200)
            # pipeline.fit(X_train, y_train)
            # y_pred = pipeline.predict(X_test)


            # 计算准确率
            accuracy = accuracy_score(y_test, y_pred)
            Accuracys.append(accuracy)
            print('Accuracy:', accuracy)
            # 计算敏感度
            if args.cls == 2:
                recall = recall_score(y_test, y_pred)
            else:
                recall = recall_score(y_test, y_pred,average=None)
            Recalls.append(recall)
            print('Recall:', recall)
            # 计算特异度
            if args.cls == 2:
                specificity = recall_score(y_test, y_pred, pos_label=0)
            else:
                specificity = recall_score(y_test, y_pred, pos_label=0,average=None) # pos_label=0表示健康
            Specificitys.append(specificity)
            print('Specificity:', specificity)
            # 输出混淆矩阵
            print('Confusion matrix:')
            print(confusion_matrix(y_test, y_pred))
            # cm = confusion_matrix(y_test, y_pred)
            # sns.heatmap(cm, annot=True)
            # plt.xlabel('Predicted labels')
            # plt.ylabel('True labels')
            # plt.title('Confusion matrix')
            # plt.savefig(args.model+'_confusion_matrix.png')
            # 对ROC曲线进行分析
            # AUC
            if accuracy > best_acc:
                best_acc = accuracy
                y_pred_proba_best = y_pred_proba
            if args.cls == 2:
                # thresholds = np.linspace(min(y_pred), max(y_pred), num=1000)

                # 计算在这些阈值下的FPR和TPR
                # fpr, tpr, thresholds= custom_roc_curve(y_test, y_pred)
                fpr, tpr, thresholds = roc_curve(y_test, y_pred)
                auc_score = auc(fpr, tpr)
                AUCs.append(auc_score)
                print('AUC:', auc_score)
                # AP
                precision, recall, th = precision_recall_curve(y_test, y_pred)
                pr_auc = auc(recall, precision)
                APs.append(pr_auc)
                print('AP:', pr_auc)
            # if specificity>0.7:print("------------------------------------------------------------------------------------------------")

            if(args.s): # SVC和LR没有feature_importances_属性
                # print('Feature importances:', pipeline.named_steps['classifier'].feature_importances_
                custom_feature_names = ['max_val', 'min_val', 'mean_val', 'var_val', 'std_val', 'slope_val', 'amplitude_val', 
                                        'midian_val', 'cv_val', 'skew_val', 
                                        'ema_max_min0', 'ema_max_min1', 'ema_max_min2', 'ema_max_min3', 'ema_max_min4', 'ema_max_min5','HP','Shetai','Stomachache']
                feature_importances = pipeline.feature_importances_
                print('Feature importances:', feature_importances)
                with open('output.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    
                    # 按行写入数据，每行10个元素
                    for i in range(0, 180, 10):
                        # 写入最多10个元素，如果不足10个，剩余的将留空
                        writer.writerow(feature_importances[i:i+10])
                # 创建条形图
                # 选择最重要的前10个特征
                indices = np.argsort(feature_importances)[::-1][:10]
                plt.figure(figsize=(10, 6))
                sns.barplot(x=feature_importances[indices], y=np.array(range(1, 11))[::-1])
                plt.title('Top 10 Feature Importances')
                plt.xlabel('Importance')
                plt.ylabel('Feature Rank')
                plt.show()
                plt.savefig(args.model+'_heatmap.png')

                # 按名字输出
                # for feature_name, feature_importance in zip(custom_feature_names, feature_importances):
                # print(feature_name, feature_importance)
        # 绘制折线图
        # plt.plot(range(10,100,10), Accuracys, label='Accuracy')
        # plt.plot(range(10,100,10), Recalls, label='Recall')
        # plt.plot(range(10,100,10), Specificitys, label='Specificity')
        # plt.plot(range(10,100,10), AUCs, label='AUC')
        # plt.plot(range(10,100,10), APs, label='AP')
        # plt.xlabel('n_estimators')
        # plt.ylabel('Value')
        # plt.legend()
        # plt.title(args.model)
        # plt.savefig(args.model+'_indexes.png')
        
        # 绘制ROC曲线
        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_best)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig(args.model+'_roc_curve.png')