import numpy as np
from matplotlib import pyplot as plt
from EGER_Visualization import Read_NPZ_Data  # 确保导入你实现的模块
from event_accumulate_frame import get_image_size

def SBT(events, img_size, num_bins, tau):
    # 初始化累积图像
    SBT = np.zeros((num_bins, img_size[0], img_size[1]))
    events['t'] = events['t'] / 1e7
    t_ref = events['t'][-1]

    # 将事件按时间分段
    max_time = events['t'][-1]
    min_time = events['t'][0]
    time_interval = (max_time - min_time) / num_bins

    for i in range(num_bins):
        start_time = min_time + i * time_interval
        end_time = start_time + time_interval
        time_mask = (events['t'] >= start_time) & (events['t'] < end_time)

        # 在当前时间段内累积事件
        for x, y, p, t in zip(events['x'][time_mask], events['y'][time_mask], events['p'][time_mask],
                              events['t'][time_mask]):
            decay_value = np.exp(-(t_ref - t) / tau)
            if p == 1:
                SBT[i, y, x] += decay_value
            else:
                SBT[i, y, x] -= decay_value

    return SBT

def visualize_SBT(SBT_img):
    num_bins, height, width = SBT_img.shape
    fig, axes = plt.subplots(1, num_bins, figsize=(15, 5))

    for i in range(num_bins):
        ax = axes[i]
        ax.imshow(SBT_img[i], cmap='gray')
        ax.set_title(f'Time Bin {i+1}')
        ax.axis('off')

    plt.show()

if __name__ == '__main__':
    # 设置文件路径和索引
    data_dir = '/home/s4090/Lyd/gem/codes/datasets/HS-ERGB/train'
    file_idx = 1  # 根据需要设置文件索引

    # 初始化数据集对象
    dataset = Read_NPZ_Data(root_path=data_dir, train=False)

    # tau = 100e-3
    tau = 50e-3  # 50ms

    num_bins = 8

    # 读取指定索引处的事件数据
    events, blur1, exp_start1, exp_end1, blur2, exp_start2, exp_end2, _, _, _ = dataset.get_data(file_idx)

    img_size = get_image_size(events)

    SBT_img = SBT(events, img_size, num_bins, tau)
    # 可视化 SBT 图像
    visualize_SBT(SBT_img)