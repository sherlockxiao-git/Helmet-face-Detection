from json import JSONDecoder
import cv2
import time
import threading
import sys
import dialog
from dialog_1 import Ui_Dialog
from PyQt5.QtWidgets import QApplication,QDialog,QLabel
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtMultimedia import (QCameraInfo,QCameraImageCapture,
	QImageEncoderSettings,QMultimedia,QVideoFrame,QSound,QCamera)



########################################
## 导入识别库
import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
from FaceDetect import FaceDetect
import face_recognition
import sqlact
#############################################

# Face++官方接口封装
result=""
class Dialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.detector = FaceDetect(0, 'facedata')
        self.encoding_list, self.name_list = self.detector.get_name_list()
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
        self.capture.imageCaptured.connect(self.do_imageCaptured)
    def accept(self):
        pass
    def reject(self):
        pass

    @pyqtSlot()
    def on_pushButton_clicked(self):#开启摄像头识别
        global result
        self.camera.stop()
        self.camera.searchAndLock()
        self.camera.unlock()
        face_name =[]
        yolo = YOLO()
        output_path = ""
        vid =cv2.VideoCapture(0,cv2.CAP_DSHOW)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        isOutput = True if output_path != "" else False
        if isOutput:
            print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        while True:
            return_value, frame = vid.read()
            image = Image.fromarray(frame)
            image = yolo.detect_image(image)
            result_1 = np.asarray(image)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0

            cv2.putText(result_1, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)


            test_locations = face_recognition.face_locations(result_1)
            test_encodings = face_recognition.face_encodings(result_1, test_locations)
            for face_encoding in test_encodings:
                face_distances = face_recognition.face_distance(self.encoding_list, face_encoding)
                best_index = np.argmin(face_distances)
                if face_distances[best_index] <= 0.55:
                    re_name = sqlact.search_by_path("face", self.name_list[best_index])
                    if yolo.label == "person":
                        sqlact.update_one_sql("face", self.name_list[best_index])
                        face_name.append(re_name[0][1])
                else:
                    face_name.append("unknown")

            for i, (top, right, bottom, left) in enumerate(test_locations):
                name = face_name[i]
                cv2.putText(result_1, name, (left + 6, bottom + 15), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)
            cv2.imshow("FaceReconition", result_1)

            # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            # cv2.imshow("result", result_1)


            # show_2 = cv2.resize(result_1, (521, 481))
            # show_3 = cv2.cvtColor(show_2, cv2.COLOR_BGR2RGB)
            # detect_image = QImage(show_3.data, show_3.shape[1], show_3.shape[0], QImage.Format_RGB888)
            # self._ui.widget(QPixmap.fromImage(detect_image))


            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyWindow("FaceReconition")
                vid.release()
                break
        yolo.close_session()
        time.sleep(2)
        view = sqlact.search_all_sql()
        re = [str(t) for i in view for t in i]
        li = ""
        for i in range(len(re)):
            if i % 2 == 0 and i != 0:
                li += "\n"
            li += re[i] + " "
        result = li
        self._ui.lineEdit.setText(result)


    def on_pushButton_2_clicked(self):#返回
        self.camera.stop()
        self.close()

    def do_imageCaptured(self,imageID,preview):
        pass

    def lineeditset(self):
        self.lineEdit.setText(result)

###################################################
# 添加摄像头所需 YOLO 对象
class YOLO(object):
    _defaults = {
        "model_path": 'trained_weights_stage_1.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/voc_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        self.label =""
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()



        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            self.label = predicted_class
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))  # 最后的标签和坐标

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print('time：', end - start)
        return image

    def close_session(self):
        self.sess.close()


############################################################


# def compareIm(faceId1, faceId2):
#     # 传送两个本地图片地址 例如："D:/Downloads/wt.jpg"
#     try:
#         # 官方给你的接口地址
#         compare_url = "https://api-cn.faceplusplus.com/facepp/v3/compare"
#         # 创建应用分配的key和secret
#         key = "MGS1NV6UEoPTxvoSTJYv8zsKv6an3cPl"
#         secret = "qAddmxSmzW_9rm8dCDsp0bVmAtrAV0Y8"
#         # 创建请求数据
#         data = {"api_key": key, "api_secret": secret}
#         files = {"image_file1": open(faceId1, "rb"), "image_file2": open(faceId2, "rb")}
#         # 通过接口发送请求
#         response = requests.post(compare_url, data=data, files=files)
#
#         req_con = response.content.decode('utf-8')
#         req_dict = JSONDecoder().decode(req_con)
#         # print(req_dict)
#         # 获得json文件里的confidence值，也就是相似度
#         confindence = req_dict['confidence']
#         if confindence > 75:
#             print("图片相似度：", confindence)
#         # confindence为相似度
#         return confindence
#     except Exception:
#         pass
#         # print("无法识别！")
#
#
# # 无限调用face++识别接口，并根据返回相似度判断人脸
# def sbdg(i):
#     for k in range(1):
#         try:
#             if compareIm(imgdict[i],"D:/qt5design/wt.jpg") > 75:
#                 print("身份确认是：", i)
#                 global result
#                 result=str(i)
#         except Exception:
#             pass
#
# #
# #
# imgdict = {"路程": "D:/python文件夹/pycharm_project/untitled2/untitled2/face_data/LuCheng.jpg","刘翔": "D:/python文件夹/pycharm_project/untitled2/untitled2/face_data/LuCheng.jpg","王自如": "D:/python文件夹/pycharm_project/untitled2/untitled2/face_data/LuCheng.jpg"}
# #
# # # 开启摄像头
# # cap = cv2.VideoCapture(0)
# # # 开启捕捉摄像头进程
# # threading.Thread(target=getimg).start()
# # # 每个匹配对象创建一个线程，为了降低等待延迟
# def RUN():
#     for x in imgdict:
#         threading.Thread(target=sbdg, args=(x,)).start()
if __name__ == "__main__":
    app = QApplication(sys.argv)
    myDialog = Dialog()
    myDialog.show()
    sys.exit(app.exec_())