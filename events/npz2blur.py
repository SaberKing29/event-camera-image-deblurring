import numpy as np
import cv2
import matplotlib.pyplot as pltz

# 读取 npz 文件
data = np.load('/home/s4090/Lyd/gem/codes/datasets/HS-ERGB/train/andle_00001.npz', allow_pickle=True)

# 获取模糊图像
blur1 = data['blur1']
blur2 = data['blur2']

# # 转换图像数据格式（将通道维度移到最后）
# blur1_img = blur1.transpose(1, 2, 0)
# blur2_img = blur2.transpose(1, 2, 0)
#
# # 拼接图像（水平拼接）
# combined_img = np.hstack((blur1_img, blur2_img))
#
# # 显示拼接后的图像
# cv2.imshow('Combined Image', combined_img)
#
# # 等待键盘输入，按任意键继续
# cv2.waitKey(0)
#
# # 保存拼接后的图像
# cv2.imwrite('combined.png', combined_img)
#
# # 销毁所有窗口
# cv2.destroyAllWindows()

# 保存第一张模糊图像为 PNG 文件
cv2.imwrite('blur1.png', blur1.squeeze().transpose(1, 2, 0))

# 保存第二张模糊图像为 PNG 文件
cv2.imwrite('blur2.png', blur2.squeeze().transpose(1, 2, 0))

# 检查 npz 文件中是否存在清晰图像
if 'sharp_imgs' in data:
    sharp_imgs = data['sharp_imgs']
    # 循环保存清晰图像
    for i in range(len(sharp_imgs)):
        sharp_img = sharp_imgs[i]
        cv2.imwrite(f'sharp_img_{i}.png', sharp_img.squeeze().transpose(1, 2, 0))
else:
    print("No sharp images found in the npz file.")
