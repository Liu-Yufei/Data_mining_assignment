import numpy as np
from scipy import stats
def get_ema_sample(data_sample, alpha): # 获取ema样本，用于特征提取
    num_channel = data_sample.shape[1]
    ema_sample = np.zeros_like(data_sample) # 生成一个和data_sample一样大小的数组
    for i in range(num_channel):
        data_v = data_sample[:, i]
        y = np.zeros_like(data_v)
        for k in range(len(y)):
            if k >= 1:
                y[k] = (1 - alpha) * y[k - 1] + alpha * (data_v[k] - data_v[k - 1])
            else:
                y[0] = alpha * data_v[0]

        ema_sample[:, i] = np.array(y)

    return ema_sample

def ge_ema(data_arr):
    # max_val = np.max(data_arr, axis=0)
    # area = np.sum(data_arr, axis=0)
    # # 对area做一个平均值
    # area = area / data_arr.shape[0]

    a = [1 / 10000, 1 / 1000, 1 / 100]
    ema_max_min = []
    for alpha in a:
        ema_sample = get_ema_sample(data_arr, alpha)
        ema_max_min.append(np.max(ema_sample, axis=0))
        ema_max_min.append(np.min(ema_sample, axis=0))

    ## logger.info('[ DataModel ] --< 特 征 >--  获取图谱特征成功')
    # logger.info('--< 特 征 >--  area shape %s' % str(area.shape))
    # logger.info('--< 特 征 >--  ema_max_min[0] shape %s' % str(ema_max_min[0].shape))
    # logger.info('--< 特 征 >--  ema_max_min[1] shape %s' % str(ema_max_min[1].shape))
    # logger.info('--< 特 征 >--  ema_max_min[2] shape %s' % str(ema_max_min[2].shape))

    new_data = [list(ema_max_min[0]), list(ema_max_min[1]), list(ema_max_min[2]),
                    list(ema_max_min[3]), list(ema_max_min[4]), list(ema_max_min[5])]

    return new_data

def statistics_feature(data_arr):

    # 最大值
    max_val = np.max(data_arr, axis=0)
    max_loc = np.argmax(data_arr, axis=0)

    # 最小值
    min_val = np.min(data_arr, axis=0)
    min_loc = np.argmin(data_arr, axis=0)

    # 均值
    mean_val = np.mean(data_arr, axis=0)

    # 方差
    var_val = np.var(data_arr, axis=0)

    # 标准差
    std_val = np.std(data_arr, axis=0)

    # 振幅
    slope_val = max_val - min_val

    # 斜率
    amplitude_val = slope_val / (max_loc - min_loc)
    amplitude_val[np.isnan(amplitude_val)] = 0

    # 曲线下面积
    # area_val = np.trapz(data_arr, axis=1)

    area_val = np.sum(data_arr, axis=0)
    sub_0_area = data_arr[0, :]
    sub_n_area = data_arr[data_arr.shape[1] - 1, :]
    area_val = area_val - (sub_0_area + sub_n_area) / 2

    # 中位数
    midian_val = np.median(data_arr, axis=0)

    # 变异系数
    cv_val = std_val / mean_val
    cv_val[np.isnan(cv_val)] = 0
    # 偏度
    skew_val = stats.skew(data_arr, axis=0)
    skew_val[np.isnan(skew_val)] = 0
    # 峰度
    kurtosis = stats.kurtosis(data_arr, axis=0)
    kurtosis[np.isnan(kurtosis)] = 0
    statistics_feature = [list(max_val), list(min_val), list(mean_val), list(var_val),
                            list(std_val), list(slope_val), list(amplitude_val), list(area_val),
                            list(midian_val), list(cv_val), list(skew_val), list(kurtosis)]
    return statistics_feature


def get_all_feature(data_arr):
    # print(data_arr)
    statistics_data = statistics_feature(data_arr)

    statistics_data = np.array(statistics_data)
    ema_data = ge_ema(data_arr)
    ema_data = np.array(ema_data)

    all_feature = np.concatenate((statistics_data,ema_data),axis=0)
    return all_feature

if __name__ == '__name__':
    ge_ema()