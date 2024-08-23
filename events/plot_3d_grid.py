import numpy as np
from matplotlib import pyplot as plt
from EGER_Visualization import Read_NPZ_Data  # 确保导入你实现的模块
from event_accumulate_frame import get_image_size

def plot_3d_grid(events, tau, img_size):
    # 将时间戳从微秒转换为秒
    events['t'] = events['t'] / 1e7
    t_ref = events['t'][-1]
    # img = np.zeros(shape=img_size, dtype=int)
    sae = np.zeros(img_size, np.float32)

    for i in range(len(events['t'])):
        decay_value = np.exp(-(t_ref - events['t'][i]) / tau)
        if (events['p'][i] == 0):
            # sae[events['y'][i], events['x'][i]] = np.exp(-(t_ref - events['t'][i]) / tau)
            sae[events['y'][i], events['x'][i]] = decay_value

    return sae

if __name__ == '__main__':
    # 设置文件路径和索引
    data_dir = '/home/s4090/Lyd/gem/codes/datasets/HS-ERGB/train'
    file_idx = 1  # 根据需要设置文件索引

    # 初始化数据集对象
    dataset = Read_NPZ_Data(root_path=data_dir, train=False)

    # tau = 100e-3
    tau = 50e-3  # 50ms

    # 读取指定索引处的事件数据
    events, blur1, exp_start1, exp_end1, blur2, exp_start2, exp_end2, _, _, _ = dataset.get_data(file_idx)

    img_size = get_image_size(events)

    time_surface_img = plot_3d_grid(events, tau, img_size)

    # select a roi, to avoid to much data.
    roi_x0 = 300
    roi_x1 = 900
    roi_y0 = 300
    roi_y1 = 900
    x_range = np.arange(roi_x0, roi_x1)
    y_range = np.arange(roi_y0, roi_y1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xx, yy = np.meshgrid(x_range, y_range)
    x, y = xx.ravel(), yy.ravel()

    top = time_surface_img[roi_y0:roi_y1, roi_x0:roi_x1].ravel()
    # 打印top数组的基本信息
    print("Top array min value:", np.min(top))
    print("Top array max value:", np.max(top))
    print("Top array mean value:", np.mean(top))
    print("Top array values:", top)
    colors = plt.cm.jet(top / np.max(top))  # color coding
    bottom = np.zeros_like(top)

    ax.bar3d(x, y, bottom, 1, 1, top, shade=True, color=colors)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('time-surface 3d grid')
    plt.show()
