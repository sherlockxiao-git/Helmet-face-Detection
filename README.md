# Helmet-face-Detection---v1.0
 helmet and face detection 
### 为解决安全头盔识别问题，采用Yolo v3算法，对数据集进行训练，训练出的结果用来检测安全帽；为解决佩戴以及未佩戴安全头盔的人脸识别问题，运用face_recognition的接口进行人脸识别
 
测试运行系统为windows10  
环境  python3.6 mysql  
必要库 tensorflow1.13 opencv pyqt5 dlib face_recognition pymysql  


# 界面
## 按钮的加入和位置修改
self.pushButton = QtWidgets.QPushButton(Dialog)
self.pushButton.setGeometry(QtCore.QRect(190, 630, 131, 41))位置修改
font = QtGui.QFont()
font.setFamily("Bahnschrift SemiBold")
font.setPointSize(20)
font.setBold(True)
font.setWeight(75)
self.pushButton.setFont(font)
self.pushButton.setObjectName("pushButton")
## 背景颜色修改
pe = QPalette()
Dialog.setAutoFillBackground(True)
pe.setColor(QPalette.Window, Qt.gray)  # 设置背景色
Dialog.setPalette(pe)
## 透明度修改
Dialog.setWindowOpacity(0.95)
## 取消边框
Dialog.setWindowFlag(QtCore.Qt.FramelessWindowHint)
# 人脸识别库
主体内容封装在
 
FaceDetect类中
 
视频检测indir输入为视频地址，摄像头检测indir输入为0
人脸识别函数
 
用于测试阶段显示，正常情况关闭
 
# 安全头盔库
主要功能已封装在
class YOLO(object):
YOLO类中
def detect_image(self, image):
检测照片中是否存在违规行为
def detect_video(yolo, video_path, output_path=""):
检测是否存在违规行为
如果需要使用摄像头
可以将
vid = cv2.VideoCapture(video_path)
传入的video_path改0
数据库
数据库已封装的操作
def delete_one_sql(tablename, image_id,id):
删除一个人的全部信息
def add_to_sql(tablename, image_id, id name):
增加一个人的信息到数据库
def update_one_sql(tablename,id, name):
增加违章记录
def search_by_path(tablename,id, path):
通过id查找该人员信息
def delete_illegal(id)
通过id删除违章记录  
# 注：
## 当前实现功能还不完善，仅提供测试，后续有时间在继续优化
