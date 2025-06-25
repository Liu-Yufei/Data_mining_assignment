import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.stats import kruskal, mannwhitneyu
def get_feature(data_arr):

    # 最大值
    max_val = np.max(data_arr, axis=0)
    max_loc = np.argmax(data_arr, axis=0)

    # 最小值
    min_val = np.min(data_arr, axis=0)
    min_loc = np.argmin(data_arr, axis=0)

    # 振幅
    slope_val = max_val - min_val
    slope_val[np.isnan(slope_val)] = 0
    # 斜率
    amplitude_val = slope_val / (max_loc - min_loc)
    amplitude_val[np.isnan(amplitude_val)] = 0
    return slope_val, amplitude_val

def calculate_mean_rank(data):
    valid_data = np.array(data)[~np.isnan(data)]
    
    if len(valid_data) == 0:
        return np.nan  # 如果去除无效值后没有有效数据，返回NaN
    
    ranks = rankdata(valid_data)
    mean_rank = np.mean(ranks)
    return mean_rank

def get_feature_5(cag_path,cag_im_path,cag_cancer_path,jk_path,cnag_path):
    slope_vals_jk = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    amplitude_vals_jk = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    slope_vals_cag = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    amplitude_vals_cag = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    slope_vals_cag_im = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    amplitude_vals_cag_im = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    slope_vals_cag_cancer = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    amplitude_vals_cag_cancer = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    slope_vals_cnag = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    amplitude_vals_cnag = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    for filename in os.listdir(jk_path):
        if os.path.isfile(os.path.join(jk_path, filename)):
            with open(os.path.join(jk_path, filename), 'r') as file:
                df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2, usecols=range(10))
                data = df.to_numpy()
                slope_val, amplitude_val = get_feature(data)
                for i in range(10):
                    # print(i)
                    slope_vals_jk[i].append(slope_val[i])
                    amplitude_vals_jk[i].append(amplitude_val[i])
    for filename in os.listdir(cag_path):
        if os.path.isfile(os.path.join(cag_path, filename)):
            with open(os.path.join(cag_path, filename), 'r') as file:
                df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2, usecols=range(10))
                data = df.to_numpy()
                slope_val, amplitude_val = get_feature(data)
                for i in range(10):
                    slope_vals_cag[i].append(slope_val[i])
                    amplitude_vals_cag[i].append(amplitude_val[i])
    for filename in os.listdir(cag_im_path):
        if os.path.isfile(os.path.join(cag_im_path, filename)):
            with open(os.path.join(cag_im_path, filename), 'r') as file:
                df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2, usecols=range(10))
                data = df.to_numpy()
                slope_val, amplitude_val = get_feature(data)
                for i in range(10):
                    slope_vals_cag_im[i].append(slope_val[i])
                    amplitude_vals_cag_im[i].append(amplitude_val[i])
    for filename in os.listdir(cag_cancer_path):
        if os.path.isfile(os.path.join(cag_cancer_path, filename)):
            with open(os.path.join(cag_cancer_path, filename), 'r') as file:
                df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2, usecols=range(10))
                data = df.to_numpy()
                slope_val, amplitude_val = get_feature(data)
                for i in range(10):
                    slope_vals_cag_cancer[i].append(slope_val[i])
                    amplitude_vals_cag_cancer[i].append(amplitude_val[i])

    for filename in os.listdir(cnag_path):
        if os.path.isfile(os.path.join(cnag_path, filename)):
            with open(os.path.join(cnag_path, filename), 'r') as file:
                df = pd.read_csv(file, header=None, sep=r'[,\s]+', engine='python',skiprows=2, usecols=range(10))
                data = df.to_numpy()
                slope_val, amplitude_val = get_feature(data)
                for i in range(10):
                    slope_vals_cnag[i].append(slope_val[i])
                    amplitude_vals_cnag[i].append(amplitude_val[i])
    return slope_vals_jk, amplitude_vals_jk, slope_vals_cag, amplitude_vals_cag, slope_vals_cag_im, amplitude_vals_cag_im, slope_vals_cag_cancer, amplitude_vals_cag_cancer, slope_vals_cnag, amplitude_vals_cnag


# 5分类
cag_path = "/home/lyf/code_cg/data_5cls/CAG"
cag_im_path = "/home/lyf/code_cg/data_5cls/CAG-IM"
cag_cancer_path = "/home/lyf/code_cg/data_5cls/CAG-cancer"
jk_path = "/home/lyf/code_cg/data_5cls/JK"
cnag_path = "/home/lyf/code_cg/data_5cls/CNAG"

# 进行Kruskal-Wallis H检验
slope_vals_jk, amplitude_vals_jk, slope_vals_cag, amplitude_vals_cag, slope_vals_cag_im, amplitude_vals_cag_im, slope_vals_cag_cancer, amplitude_vals_cag_cancer, slope_vals_cnag, amplitude_vals_cnag = get_feature_5(cag_path,cag_im_path,cag_cancer_path,jk_path,cnag_path)
stastic_slope = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
p_value_slope = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
s_s = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
p_s = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
s_a = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
p_a = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
stastic_amplitude = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
p_value_amplitude = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
mean_rank_slope = {'jk':[],'cag':[],'cag_im':[],'cag_cancer':[],'cnag':[]}
mean_rank_amplitude = {'jk':[],'cag':[],'cag_im':[],'cag_cancer':[],'cnag':[]}
for i in range(10):
    combine_slope = slope_vals_jk[i]+slope_vals_cag[i]+slope_vals_cag_im[i]+slope_vals_cag_cancer[i]+slope_vals_cnag[i]
    combine_slope_tmp = rankdata(np.array(combine_slope))
    mean_rank_slope['jk'].append(np.mean(combine_slope_tmp[:len(slope_vals_jk[i])]))
    mean_rank_slope['cag'].append(np.mean(combine_slope_tmp[len(slope_vals_jk[i]):len(slope_vals_jk[i])+len(slope_vals_cag[i])]))
    mean_rank_slope['cag_im'].append(np.mean(combine_slope_tmp[len(slope_vals_jk[i])+len(slope_vals_cag[i]):len(slope_vals_jk[i])+len(slope_vals_cag[i])+len(slope_vals_cag_im[i])]))
    mean_rank_slope['cag_cancer'].append(np.mean(combine_slope_tmp[len(slope_vals_jk[i])+len(slope_vals_cag[i])+len(slope_vals_cag_im[i]):len(slope_vals_jk[i])+len(slope_vals_cag[i])+len(slope_vals_cag_im[i])+len(slope_vals_cag_cancer[i])]))
    mean_rank_slope['cnag'].append(np.mean(combine_slope_tmp[len(slope_vals_jk[i])+len(slope_vals_cag[i])+len(slope_vals_cag_im[i])+len(slope_vals_cag_cancer[i]):]))
    
    combine_amplitude = amplitude_vals_jk[i]+amplitude_vals_cag[i]+amplitude_vals_cag_im[i]+amplitude_vals_cag_cancer[i]+amplitude_vals_cnag[i]
    combine_amplitude_tmp = rankdata(np.array(combine_amplitude))   
    mean_rank_amplitude['jk'].append(np.mean(combine_amplitude_tmp[:len(amplitude_vals_jk[i])]))
    mean_rank_amplitude['cag'].append(np.mean(combine_amplitude_tmp[len(amplitude_vals_jk[i]):len(amplitude_vals_jk[i])+len(amplitude_vals_cag[i])]))
    mean_rank_amplitude['cag_im'].append(np.mean(combine_amplitude_tmp[len(amplitude_vals_jk[i])+len(amplitude_vals_cag[i]):len(amplitude_vals_jk[i])+len(amplitude_vals_cag[i])+len(amplitude_vals_cag_im[i])]))
    mean_rank_amplitude['cag_cancer'].append(np.mean(combine_amplitude_tmp[len(amplitude_vals_jk[i])+len(amplitude_vals_cag[i])+len(amplitude_vals_cag_im[i]):len(amplitude_vals_jk[i])+len(amplitude_vals_cag[i])+len(amplitude_vals_cag_im[i])+len(amplitude_vals_cag_cancer[i])]))
    mean_rank_amplitude['cnag'].append(np.mean(combine_amplitude_tmp[len(amplitude_vals_jk[i])+len(amplitude_vals_cag[i])+len(amplitude_vals_cag_im[i])+len(amplitude_vals_cag_cancer[i]):]))

    stastic_slope[i],p_value_slope[i] = kruskal(slope_vals_jk[i], slope_vals_cag[i], slope_vals_cag_im[i], slope_vals_cag_cancer[i], slope_vals_cnag[i])
    stastic_amplitude[i],p_value_amplitude[i] = kruskal(amplitude_vals_jk[i], amplitude_vals_cag[i], amplitude_vals_cag_im[i], amplitude_vals_cag_cancer[i], amplitude_vals_cnag[i])
    s_s[i],p_s[i] = mannwhitneyu(slope_vals_jk[i], slope_vals_cag[i])
    s_a[i],p_a[i] = mannwhitneyu(amplitude_vals_jk[i], amplitude_vals_cag[i])
    print("第",i,"个传感器:")
    print('振幅(s,p):',stastic_slope[i],p_value_slope[i])
    print('斜率(s,p):',stastic_amplitude[i],p_value_amplitude[i])
    if p_value_slope[i] < 0.05:
        print('不同类之间振幅有显著性差异')
    else:
        print('不同类之间振幅没有显著性差异')
    if p_value_amplitude[i] < 0.05:
        print('不同类之间斜率有显著性差异')
    else:
        print('不同类之间斜率没有显著性差异')
    print('----------------------')
    # 
# # 计算每个组的秩均值
# mean_ranks_slope_result = {group: calculate_mean_rank(data) for group, data in slope_vals_jk.items()}
# mean_ranks_amplitude_result = {group: calculate_mean_rank(data) for group, data in amplitude_vals_jk.items()}

# print("Slope Mean Ranks:")
# for group, mean_rank in mean_ranks_slope_result.items():
#     print(f"{group}: {mean_rank}")

# print("\nAmplitude Mean Ranks:")
# for group, mean_rank in mean_ranks_amplitude_result.items():
#     print(f"{group}: {mean_rank}")
for key in mean_rank_slope.keys():
    print(key[:7],end='\t\t')
    for i in range(10):
        print(f"{mean_rank_slope[key][i]:.2f}",end=' ')
    print()
print()
    # print([mean_rank_slope[key][i] for i in range(10)])
for key in mean_rank_amplitude.keys():
    print(key[:7],end='\t\t')
    for i in range(10):
        print(f"{mean_rank_amplitude[key][i]:.2f}",end=' ')
    print()

print(i,'斜率\t\ts\t\tp')
for i in range(10):
    # 斜率的s,p:
    # s_cnag_amplitude,p_cnag_amplitude = mannwhitneyu(amplitude_vals_cag[i], amplitude_vals_cnag[i])
    # s_cagim_amplitude,p_cagim_amplitude = mannwhitneyu(amplitude_vals_cag[i], amplitude_vals_cag_im[i])
    # s_jk_cag_amplitude,p_jk_cag_amplitude= mannwhitneyu(amplitude_vals_jk[i], amplitude_vals_cag[i])
    # print(f"{s_cnag_amplitude:.2e}"+","+f"{p_cnag_amplitude:.2e}")
    # print(f"{s_cagim_amplitude:.2e}"+","+f"{p_cagim_amplitude:.2e}")
    # print(f"{s_jk_cag_amplitude:.2e}"+","+f"{p_jk_cag_amplitude:.2e}")
    # print(f"{stastic_amplitude[i]:.2e}"+","+f"{p_value_amplitude[i]:.2e}")
    print(f"{s_a[i]:.2e}"+","+f"{p_a[i]:.2e}")
print(i,'振幅\t\ts\t\tp')
for i in range(10):
    # 斜率的s,p:
    # s_cnag_amplitude,p_cnag_amplitude = mannwhitneyu(amplitude_vals_cag[i], amplitude_vals_cnag[i])
    # s_cagim_amplitude,p_cagim_amplitude = mannwhitneyu(amplitude_vals_cag[i], amplitude_vals_cag_im[i])
    # s_value_slope,p_value_slope= kruskal(slope_vals_jk[i], slope_vals_cag[i])
    # stastic_slope[i],p_value_slope[i] = kruskal(slope_vals_jk[i], slope_vals_cag[i], slope_vals_cag_im[i], slope_vals_cag_cancer[i], slope_vals_cnag[i])
    # print(f"{s_cnag_amplitude:.2e}"+","+f"{p_cnag_amplitude:.2e}")
    # print(f"{s_cagim_amplitude:.2e}"+","+f"{p_cagim_amplitude:.2e}")
    # print(f"{s_jk_cag_amplitude:.2e}"+","+f"{p_jk_cag_amplitude:.2e}")
    # print(f"{stastic_slope[i]:.2e}"+","+f"{p_value_slope[i]:.2e}")
    print(f"{s_s[i]:.2e}"+","+f"{p_s[i]:.2e}")

