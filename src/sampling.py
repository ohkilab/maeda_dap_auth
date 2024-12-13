from serial import Serial
import csv
import sys
import datetime
import time
import struct
import keyboard
import pandas as pd
import fcntl
import termios
import sys
import os

# センサデータ取得コード(https://sgrsn1711.hatenablog.com/entry/2018/02/14/204044)
# macbookと相性悪いので，繋がらなくなった時は設定からbluetoothのペアリングを解除して再接続

argv = sys.argv


class BWT901CL(Serial):
    def __init__(self, Port):
        self.myserial = super().__init__(Port, baudrate=115200, timeout=1)
        while True:
            data = super(BWT901CL, self).read(1)
            if data == b"\x55":

                print("success!")
                print(bytes(data))
                super(BWT901CL, self).read(size=10)
                break
            print("trying", data)

    def readData(self, acc_state, gyro_state, angle_state, mag_state):
        while True:
            data = super(BWT901CL, self).read(size=11)
            if not len(data) == 11:
                print("byte error:", len(data))
            if data[0] == 0x55:
                # Acceleration
                if (data[1] == 0x51) & (acc_state == 0):
                    self.accel_x = (
                        int.from_bytes(data[2:4], byteorder="little", signed=True)
                        / 32768.0
                        * 16.0
                    )
                    self.accel_y = (
                        int.from_bytes(data[4:6], byteorder="little", signed=True)
                        / 32768.0
                        * 16.0
                    )
                    self.accel_z = (
                        int.from_bytes(data[6:8], byteorder="little", signed=True)
                        / 32768.0
                        * 16.0
                    )
                    self.Temp = (
                        int.from_bytes(data[8:10], byteorder="little", signed=True)
                        / 340.0
                        + 36.25
                    )
                    acc_state += 1
                    # print('acc')

                # Angular velocity
                if (data[1] == 0x52) & (gyro_state == 0):
                    self.angular_velocity_x = (
                        int.from_bytes(data[2:4], byteorder="little", signed=True)
                        / 32768
                        * 2000
                    )
                    self.angular_velocity_y = (
                        int.from_bytes(data[4:6], byteorder="little", signed=True)
                        / 32768
                        * 2000
                    )
                    self.angular_velocity_z = (
                        int.from_bytes(data[6:8], byteorder="little", signed=True)
                        / 32768
                        * 2000
                    )
                    self.Temp = (
                        int.from_bytes(data[8:10], byteorder="little") / 340.0 + 36.25
                    )
                    gyro_state += 1
                    # print('gyro')

                # Angle
                if (data[1] == 0x53) & (angle_state == 0):
                    self.angle_x = (
                        int.from_bytes(data[2:4], byteorder="little", signed=True)
                        / 32768
                        * 180
                    )
                    self.angle_y = (
                        int.from_bytes(data[4:6], byteorder="little", signed=True)
                        / 32768
                        * 180
                    )
                    self.angle_z = (
                        int.from_bytes(data[6:8], byteorder="little", signed=True)
                        / 32768
                        * 180
                    )
                    angle_state += 1
                    # print('angle')

                # Magnetic
                if (data[1] == 0x54) & (mag_state == 0):
                    self.magnetic_x = int.from_bytes(
                        data[2:4], byteorder="little", signed=True
                    )
                    self.magnetic_y = int.from_bytes(
                        data[4:6], byteorder="little", signed=True
                    )
                    self.magnetic_z = int.from_bytes(
                        data[6:8], byteorder="little", signed=True
                    )
                    mag_state += 1
                    # print('mag')

                if (
                    (acc_state == 1)
                    & (gyro_state == 1)
                    & (angle_state == 1)
                    & (mag_state == 1)
                ):
                    self.time = datetime.datetime.now()
                    # print('できてる')
                    break
            else:
                print("UART sync error:", bytes(data))

    def getSensorData(self, acc_state, gyro_state, angle_state, mag_state):
        self.readData(acc_state, gyro_state, angle_state, mag_state)
        data = [
            self.accel_x,
            self.accel_y,
            self.accel_z,
            self.angular_velocity_x,
            self.angular_velocity_y,
            self.angular_velocity_z,
            self.magnetic_x,
            self.magnetic_y,
            self.magnetic_z,
            self.angle_x,
            self.angle_y,
            self.angle_z,
            self.time,
            argv[1],
        ]
        return data

    def getData(self):
        super(BWT901CL, self).reset_input_buffer()
        is_not_sync = True
        while is_not_sync:
            data = super(BWT901CL, self).read(size=1)
            if data == b"\x55":  # UARTの同期エラーをリカバリ
                data = super(BWT901CL, self).read(size=10)
                is_not_sync = False
                break


if __name__ == "__main__":
    if not os.path.exists("../csv"):
        os.makedirs("../csv")
    if not os.path.exists("../csv/raw_csv"):
        os.makedirs("../csv/raw_csv")
    if not os.path.exists("../csv/raw_csv/" + argv[1]):
        os.makedirs("../csv/raw_csv/" + argv[1])
    now = datetime.datetime.now()
    filename0 = "../csv/raw_csv/" + argv[1] + "/" + argv[2] + ".csv"
    # jy_sensor =  BWT901CL("/dev/cu.Witmotion01") # ls /dev/cu.*
    jy_sensor = BWT901CL("/dev/cu.HC-06")  # ls /dev/cu.*
    # jy_sensor =  BWT901CL("/dev/cu.HC-062") # ls /dev/cu.*
    num = 0
    try:
        while True:
            num += 1
            acc_state = 0
            gyro_state = 0
            angle_state = 0
            mag_state = 0
            sensor_data = jy_sensor.getSensorData(
                acc_state, gyro_state, angle_state, mag_state
            )
            if num % 50 == 0:
                print(sensor_data)
            with open(filename0, "a", newline="") as f0:
                writer0 = csv.writer(f0)
                writer0.writerow(sensor_data)

    except KeyboardInterrupt:
        print("終了します")
        jy_sensor.close()
