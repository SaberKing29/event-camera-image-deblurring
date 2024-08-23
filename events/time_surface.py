import numpy as np
from matplotlib import pyplot as plt
from EGER_Visualization import Read_NPZ_Data  # 确保导入你实现的模块
from event_accumulate_frame import get_image_size

def time_surface(events, tau, img_size):
    # 将时间戳从微秒转换为秒
    events['t'] = events['t'] / 1e7
    t_ref = events['t'][-1]
    # img = np.zeros(shape=img_size, dtype=int)
    sae = np.zeros(img_size, np.float32)

    # # 调试输出事件信息
    # print("Total number of events:", len(events['x']))
    # print("First 10 events:")
    # for i in range(10):
    #     print(f"Event {i}: x={events['x'][i]}, y={events['y'][i]}, t={events['t'][i]}, p={events['p'][i]}")

    for i in range(len(events['t'])):
        decay_value = np.exp(-(t_ref - events['t'][i]) / tau)
        if (events['p'][i] > 0):
            # sae[events['y'][i], events['x'][i]] = np.exp(-(t_ref - events['t'][i]) / tau)
            sae[events['y'][i], events['x'][i]] = decay_value
        else:
            # sae[events['y'][i], events['x'][i]] = -np.exp(-(t_ref - events['t'][i]) / tau)
            sae[events['y'][i], events['x'][i]] = -decay_value

    # # 增强对比度
    # max_abs_value = np.max(np.abs(sae))
    # if i < 10:
    #     print(sae[events['y'][i], events['x'][i]])
    #
    # sae = (sae / max_abs_value) * 127 + 128  # 归一化并映射到[0, 255]范围

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

    time_surface_img = time_surface(events, tau, img_size)

    # draw image
    fig = plt.figure()
    fig.suptitle('time_surface')
    plt.imshow(time_surface_img, cmap='gray')
    plt.xlabel("x [pixels]")
    plt.ylabel("y [pixels]")
    plt.colorbar()
    plt.savefig('event_frame.jpg')
    plt.show()