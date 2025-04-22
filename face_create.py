import face_recognition
import cv2
import os
import pickle

# 配置已知人脸数据集
known_faces = {
    "wudawang": "face_recognition/images/jianhao.jpg",
    "jiazhuo": "face_recognition/images/jiazhuo.jpg",
    "yuhui": "face_recognition/images/yuhui.jpg",
    "yihao": "face_recognition/images/yihao.jpg",
    "yianhao": "face_recognition/images/jianhao.jpg", 

}

def train_face_model():
    Encodings = []
    Names = []
    
    # 遍历已知人脸数据集
    for name, image_path in known_faces.items():
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found!")
            continue
        
        # 加载图片并提取特征
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) == 0:
            print(f"No face found in {image_path}!")
            continue
            
        encoding = face_recognition.face_encodings(image, face_locations)[0]
        Encodings.append(encoding)
        Names.append(name)
        print(f"Processed {name} successfully")

    # 将人名和编码存储为字典
    face_database = dict(zip(Names, Encodings))

    # 保存训练模型
    with open('face_recognition/face_model.pkl', 'wb') as f:
        pickle.dump(face_database, f)
    print("Model training completed")


if __name__ == "__main__":
    train_face_model()
