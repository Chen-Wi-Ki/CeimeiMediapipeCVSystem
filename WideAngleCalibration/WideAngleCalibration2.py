import cv2
import glob
import numpy as np
import pickle
import os

# 棋盤規格
BOARD_RAW = 5
BOARD_COL = 8

# 圖檔路徑
IMG_SAVE_PATH = "test_img/"
IMG_RESULT_PATH = "cal_img/"

if __name__ == '__main__':
    obj_p = np.zeros((BOARD_RAW*BOARD_COL, 3), np.float32)

    # np.mgrid[0:raw, 0:col]的shape為(2, 19, 13)轉至後為(13, 19, 2)，reshape後為(13*19, 2)
    # obj_p[:, :2]===>obj_p[:, 0] and obj_p[:, 1]
    obj_p[:, :2] = np.mgrid[0:BOARD_RAW, 0:BOARD_COL].T.reshape(-1, 2)
    # print('obj_p:', obj_p)

    obj_points = []
    img_points = []

    images = glob.glob(IMG_SAVE_PATH + '*.jpg')
    for name in images:
        print('name:', name)
        img = cv2.imread(name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 尋找角點
        ret, corners = cv2.findChessboardCorners(gray, (BOARD_RAW, BOARD_COL), None)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        sub_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        obj_points.append(obj_p)
        img_points.append(sub_corners)

        img = cv2.drawChessboardCorners(gray, (BOARD_RAW, BOARD_COL), sub_corners, ret)
        cv2.imshow('img', img)
        cv2.imwrite(IMG_RESULT_PATH + 'img'+name.split(os.sep)[-1], img)
        cv2.waitKey(500)
    #標定結果:向機內的參數矩陣,畸變細數,旋轉矩陣與平移向量
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    print('ret:', ret)
    print('mtx:', mtx)
    print('dist:', dist)
    print('rvecs:', rvecs)
    print('tvecs:', tvecs)

    # 保存參數
    cal_parameter = {'ret': ret, 'mtx': mtx, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs}
    pickle.dump(cal_parameter, open("parameter", "wb"), 0)
    print("Save successfully!")
