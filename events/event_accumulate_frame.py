import numpy as np
from matplotlib import pyplot as plt
# from .EGER_Visualization import Read_NPZ_Data  # 确保导入你实现的模块
from .EGER_Visualization import *
# 累积事件帧表示的函数
def accumulate_events(events, img_size):
    img = np.zeros(shape=img_size, dtype=int)
    for i in range(len(events['x'])):
        if 0 <= events['x'][i] < img_size[1] and 0 <= events['y'][i] < img_size[0]:  # 检查边界
            img[events['y'][i], events['x'][i]] += (2 * events['p'][i] - 1)
    return img

def get_image_size(events):
    max_x = np.max(events['x'])
    max_y = np.max(events['y'])
    # 增加一个边界值，确保图像大小能包含所有事件
    return (max_y + 10, max_x + 10)

if __name__ == '__main__':
    # 设置文件路径和索引
    data_dir = 'D:\CODE\Motion_deblurring\models\GEM-main\codes\datasets\HS-ERGB\train'
    file_idx = 1  # 根据需要设置文件索引

    # 初始化数据集对象
    dataset = Read_NPZ_Data(root_path=data_dir, train=False)

    # 读取指定索引处的事件数据
    events, blur1, exp_start1, exp_end1, blur2, exp_start2, exp_end2, _, _, _ = dataset.get_data(file_idx)

    # 打印事件信息
    print("Events x-coordinates:", events['x'])
    print("Events y-coordinates:", events['y'])
    print("Events polarities:", events['p'])
    print("Events timestamps:", events['t'])
    print("Total number of events:", len(events['x']))

    # 确定图像大小
    img_size = get_image_size(events)

    # 累积事件数据
    accumulated_img = accumulate_events(events, img_size)

    # 绘制事件累积图像
    fig = plt.figure()
    fig.suptitle('Event Accumulate Frame')
    plt.imshow(accumulated_img, cmap='gray')
    plt.xlabel("x [pixels]")
    plt.ylabel("y [pixels]")
    plt.colorbar()
    plt.savefig('event_accumulate_frame.jpg')
    plt.show()
