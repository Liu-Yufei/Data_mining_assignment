import pandas as pd

import shutil
import re
path_source = 'stomachache.csv'
path_distance_train = '/home/lyf/Data/CAG_CNAG/Conv_HP/train.csv'
path_distance_test = '/home/lyf/Data/CAG_CNAG/Conv_HP/test.csv'
df_source = pd.read_csv(path_source)
df_distance_train = pd.read_csv(path_distance_train)
df_distance_test = pd.read_csv(path_distance_test)
not_found = {'0': [], '1': []}
# 遍历df_source的第一列
for index, name in df_source['腹痛-'].items():
    file_found = False
    for index2, name2 in df_distance_train['0'].items():
        # 如果name2中包含name的字符：
        if name in name2:
            file_found = True
            # 在df_distance_train['hp']中写入1
            df_distance_train.loc[index2, 'stomachache'] = '0'
            break
    for index3, name3 in df_distance_test['0'].items():
        if name in name3:
            file_found = True
            df_distance_test.loc[index3, 'stomachache'] = '0'
            break
    if not file_found:
        not_found['0'].append(name)
for index, name in df_source['腹痛+'].items():
    file_found = False
    for index2, name2 in df_distance_train['0'].items():
        # 如果name2中包含name的字符：
        if  str(name) in name2:
            file_found = True
            # 在df_distance_train['hp']中写入1
            df_distance_train.loc[index2, 'stomachache'] = '1'
            break
    for index3, name3 in df_distance_test['0'].items():
        if str(name) in name3:
            file_found = True
            df_distance_test.loc[index3, 'stomachache'] = '1'
            break
    if not file_found:
        not_found['1'].append(name)
not_found_df = pd.DataFrame({col: pd.Series(names) for col, names in not_found.items()})
not_found_file_path = '/home/lyf/code_cg/not_found_files_1.xlsx'  # 请修改为实际路径
not_found_df.to_csv(not_found_file_path, index=False)
df_distance_train.to_csv(path_distance_train, index=False)
df_distance_test.to_csv(path_distance_test, index=False)    
