import os
import numpy as np
import sys
import matplotlib.pyplot as plt

# from codes.data.utils import event2frame_accumulate

# sys.path.append('/home/s4090/Lyd/gem/codes')
# from data import utils
from ..data import utils


class Read_NPZ_Data:
    def __init__(self, root_path, num_bins=16, roi_size=(128, 128), scale_factor=4, has_gt=True, train=True,
                 predict_ts=None):
        self.__dict__.update(locals())
        self.train = train
        self.num_bins = num_bins
        self.scale_factor = scale_factor
        self.ev_roi_size = roi_size
        self.im_roi_size = (roi_size[0] * scale_factor, roi_size[1] * scale_factor)
        self.has_gt = has_gt
        self.predict_ts = predict_ts
        self.check_files(root_path)

    def get_filename(self, path, suffix):
        namelist = []
        filelist = os.listdir(path)
        for i in filelist:
            if os.path.splitext(i)[1] == suffix:
                namelist.append(i)
        namelist.sort()
        return namelist

    def check_files(self, root_path):
        self.data_path = root_path  # 修正路径为根路径
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"The path {self.data_path} does not exist.")
        self.data_list = self.get_filename(self.data_path, '.npz')

    def get_data(self, idx):
        data = np.load(os.path.join(self.data_path, self.data_list[idx]), allow_pickle=True)
        events = data['events'].item()
        blur1 = data['blur1'].astype(float)
        exp_start1 = data['exp_start1']
        exp_end1 = data['exp_end1']
        blur2 = data['blur2'].astype(float)
        exp_start2 = data['exp_start2']
        exp_end2 = data['exp_end2']
        prefix = os.path.splitext(self.data_list[idx])[0]

        # 输出总的事件数量
        total_events = len(events['x'])
        print(f"Total number of events: {total_events}")

        if self.has_gt and 'sharp_imgs' in data:
            sharp_imgs = data['sharp_imgs']
            timestamps = data['sharp_timestamps']
            print(f"Index {idx}: 有清晰图像。")
            return events, blur1, exp_start1, exp_end1, blur2, exp_start2, exp_end2, timestamps, sharp_imgs, prefix
        else:
            print(f"Index {idx}: 没有清晰图像。")
            return events, blur1, exp_start1, exp_end1, blur2, exp_start2, exp_end2, None, None, prefix


def visualize_eger(data_dir, file_idx, num_bins=16, roi_size=(128, 128), scale_factor=4):
    # 初始化数据集对象
    dataset = Read_NPZ_Data(root_path=data_dir, train=False)

    # 读取指定索引处的事件数据
    events, blur1, exp_start1, exp_end1, blur2, exp_start2, exp_end2, _, _, _ = dataset.get_data(file_idx)

    # 打印事件数据的最大和最小坐标值
    print("Max x:", np.max(events['x']), "Min x:", np.min(events['x']))
    print("Max y:", np.max(events['y']), "Min y:", np.min(events['y']))

    sharp_ts = (exp_start1 + exp_end1) // 2
    sharp_span = (sharp_ts, sharp_ts)
    blur_span = (exp_start1, exp_end2)

    # 动态调整ROI，覆盖整个图像范围
    max_x, max_y = 1440, 1080  # 根据事件的最大范围设置
    eger = utils.gen_EGER(events, num_bins, sharp_span, blur_span, roiTL=(0, 0), roi_size=(max_y, max_x))
    E1, E2, E3 = utils.gen_EGER_E1_E2_E3(events, num_bins, sharp_span, blur_span, roi_size=roi_size)

    eger_blur2sharp = utils.gen_EGER(events, num_bins, sharp_span, blur_span, roiTL=(0, 0), roi_size=(max_y, max_x))
    eger_largeblur2sharp = utils.gen_EGER(events, num_bins, sharp_span, (exp_start1, exp_end2), roiTL=(0, 0),
                                          roi_size=(max_y, max_x))
    eger_largeblur2blur = utils.gen_EGER(events, num_bins, blur_span, (exp_start1, exp_end2), roiTL=(0, 0),
                                         roi_size=(max_y, max_x))
    # 打印 eger_blur2sharp 的 shape
    print("eger_blur2sharp shape:", eger_blur2sharp.shape)
    accumulated_img_eger = utils.eger2frame_accumulate(eger_blur2sharp)
    print("accumulated_img_eger shape:", accumulated_img_eger.shape)
    # 绘制事件累积图像
    fig = plt.figure()
    fig.suptitle('Event Accumulate Frame')
    plt.imshow(accumulated_img_eger, cmap='gray')
    plt.xlabel("x [pixels]")
    plt.ylabel("y [pixels]")
    plt.colorbar()
    plt.savefig('event_accumulate_frame.jpg')
    plt.show()

    # 显示第一幅模糊图像
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(blur1.transpose(1, 2, 0) / 255.0)  # 转置维度并归一化
    plt.title('Blur1 Image')

    # 显示第二幅模糊图像
    plt.subplot(1, 2, 2)
    plt.imshow(blur2.transpose(1, 2, 0) / 255.0)  # 转置维度并归一化
    plt.title('Blur2 Image')

    plt.show()

    # 显示EGER事件表示
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(eger[0], cmap='gray')
    plt.title('EGER Visualization')

    # 显示第一幅模糊图像
    plt.subplot(1, 3, 2)
    plt.imshow(blur1.transpose(1, 2, 0) / 255.0)  # 转置维度并归一化
    plt.title('Blur1 Image')

    # 显示第二幅模糊图像
    plt.subplot(1, 3, 3)
    plt.imshow(blur2.transpose(1, 2, 0) / 255.0)  # 转置维度并归一化
    plt.title('Blur2 Image')

    plt.show()

    # 可视化 E1
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(E1[0, 0], cmap='gray')
    plt.title('E1 Visualization')

    # 可视化 E2
    plt.subplot(1, 3, 2)
    plt.imshow(E2[0, 0], cmap='gray')
    plt.title('E2 Visualization')

    # 可视化 E3
    plt.subplot(1, 3, 3)
    plt.imshow(E3[0, 0], cmap='gray')
    plt.title('E3 Visualization')

    plt.show()

    print(f"E1 shape: {E1.shape}, sum: {np.sum(E1)}")  # 添加调试打印
    print(f"E2 shape: {E2.shape}, sum: {np.sum(E2)}")  # 添加调试打印
    print(f"E3 shape: {E3.shape}, sum: {np.sum(E3)}")  # 添加调试打印
    print(f"eger shape: {eger.shape}, sum: {np.sum(eger)}")  # 添加调试打印
    # # 打印读取的数据
    # print("Events:", events)
    # print("Blur1 Image:", blur1)
    # print("Exposure Start 1:", exp_start1)
    # print("Exposure End 1:", exp_end1)
    # print("Blur2 Image:", blur2)
    # print("Exposure Start 2:", exp_start2)
    # print("Exposure End 2:", exp_end2)


if __name__ == "__main__":
    data_dir = '/home/s4090/Lyd/gem/codes/datasets/HS-ERGB/train'  # 确保路径正确
    file_idx = 1  # 根据需要设置文件索引
    visualize_eger(data_dir, file_idx)
