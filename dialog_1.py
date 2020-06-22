# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialog.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPalette, QBrush, QPixmap
import sys

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(979, 766)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(410, 40, 221, 41))
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QTextEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(630, 195, 261, 351))#(60, 110, 521, 481))
        self.lineEdit.setObjectName("TextEdit")

        self.lineEdit_1 = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_1.setGeometry(QtCore.QRect(320, 580, 300, 30))
        self.lineEdit_1.setObjectName("lineEdit_1")



        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(720, 155, 91, 21))
        font = QtGui.QFont()
        font.setFamily("Arial Narrow")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(100, 630, 200, 60))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiBold")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(650, 630, 200, 60))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiBold")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        # self.widget = QtWidgets.QWidget(Dialog)

        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setGeometry(QtCore.QRect(370, 630, 200, 60))
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiBold")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")

        self.widget = QCameraViewfinder(Dialog)
        self.widget.setGeometry(QtCore.QRect(60, 110, 521, 481))
        self.widget.setObjectName("widget")

        self.retranslateUi(Dialog)
        self.pushButton_2.clicked.connect(Dialog.close)
        # self.pushButton_3.clicked.connect(Dialog.close)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "安全头盔检测系统"))
        self.label_2.setText(_translate("Dialog", "违规名单"))
        self.pushButton.setText(_translate("Dialog", "摄像头检测"))
        self.pushButton_2.setText(_translate("Dialog", "退出"))
        self.pushButton_3.setText(_translate("Dialog", "视频检测"))
        Dialog.setWindowOpacity(0.95)
        Dialog.setWindowFlag(QtCore.Qt.WindowMinimizeButtonHint)

        pe = QPalette()
        Dialog.setAutoFillBackground(True)
        pe.setColor(QPalette.Window, Qt.gray)  # 设置背景色
        Dialog.setPalette(pe)


from PyQt5.QtMultimediaWidgets import QCameraViewfinder