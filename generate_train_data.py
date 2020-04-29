import cv2
import os
import numpy as np



def generate_positive_set(img, points, img_f):

    h, w, _ = img.shape

    samples = points[::int(np.ceil(len(points)/343.0))]
    nums = len(samples)

    for i, s in enumerate(samples):

        area = np.zeros((16, 16, 3), np.uint8)

        left_x = s[0] - 7 if s[0] - 7 >=0 else 0
        left_y = s[1] - 7 if s[1] - 7 >=0 else 0
        right_x = s[0] + 8 if s[0] + 8 < h else h-1
        right_y = s[1] + 8 if s[1] + 8 < w else w-1

        crop = img[left_x:right_x+1, left_y:right_y+1]

        area[7-(s[0]-left_x):7+(right_x-s[0])+1, 7-(s[1]-left_y):7+(right_y-s[1])+1] = crop

        # print(area.shape)
        # cv2.imshow("image", area)
        # cv2.waitKey(0)

        cv2.imwrite(os.path.join("./data", "train", "positive", os.path.splitext(img_f)[0]+"_"+str(i)+".png"), area)

    return nums



def generate_negitive_set(img, mask, img_f, num):

    h, w, _ = img.shape

    for i in range(num):

        cen_x = np.random.randint(0, h)
        cen_y = np.random.randint(0, w)

        while mask[cen_x, cen_y] == 255:
            cen_x = np.random.randint(0, h)
            cen_y = np.random.randint(0, w)

        area = np.zeros((16, 16, 3), np.uint8)

        left_x = cen_x - 7 if cen_x - 7 >= 0 else 0
        left_y = cen_y - 7 if cen_y - 7 >= 0 else 0
        right_x = cen_x + 8 if cen_x + 8 < h else h - 1
        right_y = cen_y + 8 if cen_y + 8 < w else w - 1

        crop = img[left_x:right_x + 1, left_y:right_y + 1]

        area[7 - (cen_x - left_x): 7 + (right_x - cen_x)+1, 7 - (cen_y - left_y):7 + (right_y - cen_y)+1] = crop

        cv2.imwrite(os.path.join("./data", "train", "negitive", os.path.splitext(img_f)[0] + "_" + str(i) + ".png"),
                    area)



def mask_to_skyline(mask):
    h, w = mask.shape
    points = []

    for i in range(w):  # 按列扫描
        for j in range(h):
            if mask[j][i] == 255:
                points.append([j, i])  # (高, 宽)

    print(len(points))

    return points




if __name__ == '__main__':

    data_path = "./data/"

    img_files = os.listdir(os.path.join(data_path, "image"))

    for i, img_f in enumerate(img_files):
        print(i, img_f)

        img = cv2.imread(os.path.join(data_path, "image", img_f))
        mask = cv2.imread(os.path.join(data_path, "label", img_f.replace("new", "")), 0)

        points = mask_to_skyline(mask)
        nums = generate_positive_set(img, points, img_f)

        generate_negitive_set(img, mask, img_f, nums)
