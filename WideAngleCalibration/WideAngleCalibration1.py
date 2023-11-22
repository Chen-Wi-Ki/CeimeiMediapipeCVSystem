import cv2

# 图片保存路径
IMG_SAVE_PATH = "test_img/"


if __name__ == '__main__':
    num = 1
    camera = cv2.VideoCapture(0)
    camera.set(3,320)
    camera.set(4,480)
    camera.set(5,8)
    while True:
        state, src = camera.read()
        cv2.imshow('src', src)

        if cv2.waitKey(10) & 0xff == ord(' '):
            cv2.imwrite(IMG_SAVE_PATH + str(num) + '.jpg', src)
            print("Saved img_" + str(num) + "!")
            num += 1
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()
