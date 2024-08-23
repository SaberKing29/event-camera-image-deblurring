import numpy as np
from matplotlib import pyplot as plt
from EGER_Visualization import Read_NPZ_Data  # 确保导入你实现的模块
from event_accumulate_frame import get_image_size

def event_frame(events, img_size):
    img = np.zeros(shape=img_size, dtype=int)
    for i in range(len(events['x'])):
        img[events['y'][i], events['x'][i]] = (2 * events['p'][i] - 1)

    return img

if __name__ == '__main__':
    # 设置文件路径和索引
    data_dir = '/home/s4090/Lyd/gem/codes/datasets/HS-ERGB/train'
    file_idx = 1  # 根据需要设置文件索引

    # 初始化数据集对象
    dataset = Read_NPZ_Data(root_path=data_dir, train=False)

    # 读取指定索引处的事件数据
    events, blur1, exp_start1, exp_end1, blur2, exp_start2, exp_end2, _, _, _ = dataset.get_data(file_idx)

    img_size = get_image_size(events)

    event_frame = event_frame(events,img_size)

    # draw image
    fig = plt.figure()
    fig.suptitle('Event Frame')
    plt.imshow(event_frame, cmap='gray')
    plt.xlabel("x [pixels]")
    plt.ylabel("y [pixels]")
    plt.colorbar()
    plt.savefig('event_frame.jpg')
    plt.show()
