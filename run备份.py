
from json import JSONDecoder
import cv2
import time
import threading
import sys
import dialog_1
from dialog_1 import Ui_Dialog
from PyQt5.QtWidgets import QApplication,QDialog,QLabel
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtMultimedia import (QCameraInfo,QCameraImageCapture,
	QImageEncoderSettings,QMultimedia,QVideoFrame,QSound,QCamera)
# Face++官方接口封装
result="苏玲娅"
class Dialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._ui = Ui_Dialog()
        self._ui.setupUi(self)
        # self._ui.lineEdit.setText("路程")

        self.camera = None
        cameras = QCameraInfo.availableCameras()
        if len(cameras) > 0:
            self.__initCamera()
            self.__initImageCapture()
            self.camera.start()

    def __initCamera(self):
        camInfo = QCameraInfo.defaultCamera()
        self.camera = QCamera(camInfo)
        self.camera.setViewfinder(self._ui.widget)
        self.camera.setCaptureMode(QCamera.CaptureStillImage)


    def __initImageCapture(self):
        self.capture = QCameraImageCapture(self.camera)
        setting = QImageEncoderSettings()
        setting.setCodec("image/jpeg")
        self.capture.setEncodingSettings(setting)
        self.capture.setBufferFormat(QVideoFrame.Format_Jpeg)
        self.capture.setCaptureDestination(QCameraImageCapture.CaptureToFile)
        self.capture.capture(file="D:/qt5design/wt.jpg")

    def accept(self):
        pass
    def reject(self):
        pass

    @pyqtSlot()
    def on_pushButton_clicked(self):#开启摄像头识别
        global result
        self.camera.stop()
        self.camera.searchAndLock()
        self.capture.capture(file="D:/qt5design/wt.jpg")
        self.camera.unlock()
        # RUN()
        # img = "facedata/gakki1.jpg"
        # img = cv2.imread(img)
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        success, frame = camera.read()
        while cv2.waitKey(10) == -1:
            cv2.imshow("FaceReconition", frame)
            success, frame = camera.read()
        cv2.destroyAllWindows()
        camera.release()
        time.sleep(2)
        result = "hehe"
        self._ui.lineEdit.setText(result)
    def on_pushButton_2_clicked(self):#返回
        self.camera.stop()
        self.close()

    def do_imageCaptured(self,imageID,preview):
        pass

    def lineeditset(self):
        self.lineEdit.setText(result)



def compareIm(faceId1, faceId2):
    # 传送两个本地图片地址 例如："D:/Downloads/wt.jpg"
    try:
        # 官方给你的接口地址
        compare_url = "https://api-cn.faceplusplus.com/facepp/v3/compare"
        # 创建应用分配的key和secret
        key = "MGS1NV6UEoPTxvoSTJYv8zsKv6an3cPl"
        secret = "qAddmxSmzW_9rm8dCDsp0bVmAtrAV0Y8"
        # 创建请求数据
        data = {"api_key": key, "api_secret": secret}
        files = {"image_file1": open(faceId1, "rb"), "image_file2": open(faceId2, "rb")}
        # 通过接口发送请求
        response = requests.post(compare_url, data=data, files=files)

        req_con = response.content.decode('utf-8')
        req_dict = JSONDecoder().decode(req_con)
        # print(req_dict)
        # 获得json文件里的confidence值，也就是相似度
        confindence = req_dict['confidence']
        if confindence > 75:
            print("图片相似度：", confindence)
        # confindence为相似度
        return confindence
    except Exception:
        pass
        # print("无法识别！")


# 无限调用face++识别接口，并根据返回相似度判断人脸
def sbdg(i):
    for k in range(1):
        try:
            if compareIm(imgdict[i],"D:/qt5design/wt.jpg") > 75:
                print("身份确认是：", i)
                global result
                result=str(i)
        except Exception:
            pass

#
#
imgdict = {"路程": "D:/python文件夹/pycharm_project/untitled2/untitled2/face_data/LuCheng.jpg","刘翔": "D:/python文件夹/pycharm_project/untitled2/untitled2/face_data/LuCheng.jpg","王自如": "D:/python文件夹/pycharm_project/untitled2/untitled2/face_data/LuCheng.jpg"}
#
# # 开启摄像头
# cap = cv2.VideoCapture(0)
# # 开启捕捉摄像头进程
# threading.Thread(target=getimg).start()
# # 每个匹配对象创建一个线程，为了降低等待延迟
def RUN():
    for x in imgdict:
        threading.Thread(target=sbdg, args=(x,)).start()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myDialog = Dialog()
    myDialog.show()
    sys.exit(app.exec_())