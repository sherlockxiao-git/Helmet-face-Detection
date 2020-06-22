import os
import cv2
import face_recognition
import numpy as np
import sqlact


class FaceDetect(object):
    def __init__(self, indir, facedata):
        self.indir = indir
        self.facedata = facedata



    def add_name_list(self):    #初始数据库 通过文件夹的方式添加数据
        encoding_list = []
        name_list = []
        for root, dirs, files in os.walk(self.facedata):
            for file in files:
                img = face_recognition.load_image_file(os.path.join(root, file))
                encoding = face_recognition.face_encodings(img)[0]
                encoding_list.append(encoding)
                name_list.append(os.path.splitext(file)[0])
                sqlact.add_to_sql("face", os.path.splitext(file)[0], os.path.splitext(file)[0][:-1])

        return encoding_list, name_list


    def get_name_list(self):  # 初始数据库 通过文件夹的方式添加数据
        encoding_list = []
        name_list = []
        for root, dirs, files in os.walk(self.facedata):
            for file in files:
                img = face_recognition.load_image_file(os.path.join(root, file))
                encoding = face_recognition.face_encodings(img)[0]
                encoding_list.append(encoding)
                name_list.append(os.path.splitext(file)[0])

        return encoding_list, name_list

    def face_detect(self, encoding_list, name_list):
        test_encodings = []
        face_name = []
        camera = cv2.VideoCapture(self.indir)
        success, frame = camera.read()
        while success and cv2.waitKey(10) == -1:
            test_locations = face_recognition.face_locations(frame)
            test_encodings = face_recognition.face_encodings(frame, test_locations)
            for face_encoding in test_encodings:
                face_distances = face_recognition.face_distance(encoding_list, face_encoding)
                best_index = np.argmin(face_distances)
                if face_distances[best_index] <= 0.55:
                    re_name = sqlact.search_by_path("face",name_list[best_index])
                    face_name.append(re_name[0][1])
                else:
                    face_name.append("unknown")

            for i, (top, right, bottom, left) in enumerate(test_locations):
                name = face_name[i]
                # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # cv2.rectangle(frame, (left, bottom), (right, bottom + 40), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom + 15), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)

            cv2.imshow("FaceReconition", frame)
            success, frame = camera.read()

        cv2.destroyAllWindows()
        camera.release()
        return face_name


# if __name__ == '__main__':
#     detector = FaceDetect(0, 'facedata')
#     encoding_list, name_list = detector.add_name_list()
#     face_name = detector.face_detect(encoding_list, name_list)
#     print(face_name)
    # re=sqlact.search_by_path("face","baihezi1")
    # print(re[0][1])
# sqlact.update_one_sql("face","wxl1")
# view = sqlact.search_all_sql()
# # print(",".join([str(t) for i in view for t in i ]))
# re = [str(t) for i in view for t in i ]
# li=""
# for i in range(len(re)):
#     if i%2==0 and i!=0:
#         li+="\n"
#     li+=re[i]+" "
#
# print(li)