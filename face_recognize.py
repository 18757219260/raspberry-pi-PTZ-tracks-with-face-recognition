import cv2
import face_recognition
import numpy as np
import time
from adafruit_servokit import ServoKit
import busio
import board
import pickle
import subprocess
import signal
import os

class FaceRecognizer:
    def __init__(self):
        self.known_faces = {}
        self.process = None
        self.kit = None  
        self.start_time = time.time()  # 记录程序开始时间
        self.last_move_time = 0  # 记录上次云台移动时间
        self.pan = 90  # 云台初始中立角度
        self.tilt = 90
        self.prev_error_x = 0  # PID控制变量
        self.prev_error_y = 0
        self.integral_x = 0
        self.integral_y = 0
        self.Kp, self.Ki, self.Kd = 0.04, 0.001, 0.05  # PID参数
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)  
        self._load_known_faces()
        self._init_servo_kit()
        self._init_camera()
        self.last_center_x = 160
        self.last_center_y = 120
      
    def _load_known_faces(self):
        """加载人脸数据库"""
        try:
            with open("/home/joe/chatbox/face_recognition/face_model.pkl", "rb") as f:
                self.known_faces = pickle.load(f)
                print("成功加载人脸数据库！")
        except FileNotFoundError:
            print("未找到人脸数据库，请先运行创建数据库脚本！")
            exit()

    def _init_camera(self):
        """初始化摄像头，使用libcamera-vid"""
        cmd = "libcamera-vid -t 0 --width 320 --height 240 --codec mjpeg --nopreview -o -"
        try:
            self.process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, bufsize=1024*1024)
            print("摄像头初始化成功")
        except Exception as e:
            print(f"初始化摄像头失败: {e}")
            exit()

    def _init_servo_kit(self):
        """初始化ServoKit用于云台控制"""
        try:
            self.kit = ServoKit(channels=16)
            self.reset_servos()
            print("ServoKit初始化成功")
        except Exception as e:
            print(f"初始化ServoKit失败: {e}")
            print("请检查I2C连接（SDA: GPIO 2, SCL: GPIO 3）、电源或设备地址")
            exit()

    def reset_servos(self):
        """将云台重置到中立位置"""
        self.pan = 90
        self.tilt = 90
        try:
            if self.kit:
                self.kit.servo[0].angle = self.pan  # 水平舵机
                self.kit.servo[1].angle = self.tilt  # 垂直舵机
                # print("云台已重置到中立位置（90°, 90°）")
        except Exception as e:
            print(f"重置舵机失败: {e}")

    def get_frame(self):
        """从libcamera-vid获取视频帧"""
        buffer = bytearray()
        try:
            while True:
                data = self.process.stdout.read(1024)
                if not data:
                    print("无法读取视频数据")
                    return False, None
                buffer.extend(data)
                start = buffer.find(b'\xff\xd8')
                end = buffer.find(b'\xff\xd9', start)
                if start != -1 and end != -1:
                    jpeg = buffer[start:end + 2]
                    buffer = buffer[end + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        print("解码帧失败")
                        continue
                    frame = cv2.flip(frame, 0)
                    return True, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"获取视频帧失败: {e}")
            return False, None

    def recognize_faces(self, frame):
        """执行人脸识别"""
        try:
            face_locations = face_recognition.face_locations(frame, model="hog", number_of_times_to_upsample=0)
            face_encodings = face_recognition.face_encodings(frame, face_locations, model="large")
            results = []
            start = time.time()
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(
                    list(self.known_faces.values()), face_encoding, tolerance=0.45
                )
                name = "who?"
                face_distances = face_recognition.face_distance(
                    list(self.known_faces.values()), face_encoding
                )
                if True in matches:
                    best_match_index = face_distances.argmin()
                    if face_distances[best_match_index] < 0.4:
                        name = list(self.known_faces.keys())[best_match_index]
                        end = time.time()
                        elapsed = end - start
                        # print(f"识别到你啦 {name}! 耗时 {elapsed:.3f} 秒")
                results.append(((top, right, bottom, left), name))
            # print(f"检测到 {len(results)} 张人脸")
            return results
        except Exception as e:
            print(f"人脸识别失败: {e}")
            return []

    def move_pan_tilt(self, face_location):
        """使用PID算法控制云台跟随人脸移动"""
        current_time = time.time()
        if current_time - self.last_move_time < 0.15:  
            return

        try:
            center_x = (face_location[1] + face_location[3]) // 2  # right + left
            center_y = (face_location[0] + face_location[2]) // 2  # top + bottom

            self.last_center_x = 0.8 * self.last_center_x + 0.2 * center_x
            self.last_center_y = 0.8 * self.last_center_y + 0.2 * center_y

            error_x = self.last_center_x - 160
            error_y = self.last_center_y - 120


            if abs(error_x) < 27 and abs(error_y) < 27:
                return
            

            self.integral_x += error_x
            self.integral_y += error_y
            self.integral_x = max(min(self.integral_x, 50), -50)
            self.integral_y = max(min(self.integral_y, 50), -50)

            derivative_x = error_x - self.prev_error_x
            derivative_y = error_y - self.prev_error_y


            pan_adjust = -(self.Kp * error_x + self.Ki * self.integral_x + self.Kd * derivative_x)
            tilt_adjust = -(self.Kp * error_y + self.Ki * self.integral_y + self.Kd * derivative_y)

            pan_adjust = max(min(pan_adjust, 5), -5)
            tilt_adjust = max(min(tilt_adjust, 5), -5)


            # 更新舵机角度
            self.pan -= int(pan_adjust)
            self.tilt -= int(tilt_adjust)
            self.pan = max(min(self.pan, 180), 0)
            self.tilt = max(min(self.tilt, 180), 0)

            self.kit.servo[0].angle = self.pan
            self.kit.servo[1].angle = self.tilt
            print(f"云台移动到水平: {self.pan}°, 垂直: {self.tilt}°")

            self.prev_error_x = error_x
            self.prev_error_y = error_y
            self.last_move_time = current_time

            if abs(error_x) < 5:
                self.integral_x = 0
            if abs(error_y) < 5:
                self.integral_y = 0


        except Exception as e:
            print(f"云台移动失败: {e}")

    def release(self):
        """释放资源并确保云台和I2C停止"""
        try:
            self.reset_servos()
            if self.kit:
                # 释放ServoKit的I2C资源
                for i in range(16):
                    try:
                        self.kit.servo[i].angle = None  # 断开舵机控制
                    except:
                        pass
                self.kit._pca.deinit()  # 释放底层PCA9685的I2C连接
                self.kit = None
                print("ServoKit I2C资源已释放")
            if self.process:
                self.process.terminate()
                self.process.wait(timeout=2)
                print("摄像头进程已终止")
        except Exception as e:
            print(f"释放资源失败: {e}")
        cv2.destroyAllWindows()
        print("所有资源已释放")
        
    def main(self,save_path):#主函数
        try:

            while True:
                # if time.time() - recognizer.start_time > 10:
                #     print("运行10秒后自动关闭")
                #     break

                ret, frame = self.get_frame()
                if not ret:
                    print("无法获取视频帧")
                    break

                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                results = self.recognize_faces(frame)
                # face_detected = False
                # if len(results) == 0:  
                #     print("无人，回归90度")
                #     self.reset_servos()
                #     continue


                for (top, right, bottom, left), name in results:
                    color = (0, 255, 0) if name != "who?" else (0, 0, 255)
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(display_frame, name, (left, top-15), cv2.FONT_HERSHEY_SIMPLEX,1, color, 3)
                    # if name != "who?":
                    #     face_detected = True
                    self.move_pan_tilt((top, right, bottom, left))


               
                    cv2.imwrite(save_path, display_frame)
                    # with open(save_path,'wb') as f:
                    #     pickle.dump(display_frame, f)
        except KeyboardInterrupt:
            print("用户中断程序")
        except Exception as e:
            print(f"程序异常: {e}")
        finally:
            recognizer.release()

        print("程序已结束")




if __name__ == "__main__":
    save_path = f"frame.jpg"
    recognizer = FaceRecognizer()
    recognizer.main(save_path)
           


