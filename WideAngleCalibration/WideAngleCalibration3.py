import cv2
import pickle

if __name__ == '__main__':
    # 獲取校正參數
    f = pickle.load(open('parameter', 'rb'))  # 讀取校正參數
    ret, mtx, dist, rvecs, tvecs = f['ret'], f['mtx'], f['dist'], f['rvecs'], f['tvecs']

    # 獲取圖片尺寸
    img = cv2.imread('test_img/1.jpg')
    h, w = img.shape[:2]
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    x, y, w, h = roi      # roi 若提取的不准卻可能需要手動調整
    vid = cv2.VideoCapture(0)
    vid.set(3,320)
    vid.set(4,480)
    vid.set(5,8)

    while True:
        state, src = vid.read()
        cv2.imshow('src', src)
        dst = cv2.undistort(src, mtx, dist, None, new_camera_mtx)
        cv2.imshow('img1', dst)
        dst = dst[y:y + h, x:x + w]
        cv2.imshow('img2', dst)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
