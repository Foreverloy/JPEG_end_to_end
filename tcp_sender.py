import socket
from crc32 import verify_crc32, crc32
from convolutional_coding import Convolution
from jepg import compress, decompress
from hexString_to_bitList import get_bitList_from_hexString, get_hexString_from_bitList,get_bytes_from_binaryarray
import cv2
import threading
import base64
import struct
import time
import math
import numpy as np

# 发送端函数
def start_sender(img_path: str, k: int, packet_size: int, time_out: int = 5):
    # 等待接收端启动
    time.sleep(1)
    # 创建TCP套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 10000)
    sock.connect(server_address)
    try:
        send_image(sock, img_path, k, packet_size, time_out)
    finally:
        sock.close()

# 发送数据函数
def send_image(sock, img_path, k: int, packet_size: int, timeout):
    print("Sending image...")
    # 读取图像数据
    img_data = cv2.imread(img_path, -1)
    # 得到压缩后图像数据
    img_compress = compress(img_data, 50)
    # 得到解压缩后图像数据的二进制列表形式
    img_compress = get_bitList_from_hexString(img_compress)
    # 实例化一个(2,1,k)卷积编码器，2 <= k <= 8
    conv = Convolution(k)
    send_data = []  # 待发送的数据包队列
    send_data_length = math.ceil(len(img_compress) / 8 / packet_size)  # 待发送数据包队列长度
    for i in range(send_data_length - 1):
        send_bag = img_compress[i * packet_size * 8:(i + 1) * packet_size * 8]  # 取出一个待发送队列的数据包
        send_bag_crc32 = crc32(send_bag)  # 对待发送的数据包进行crc32编码
        send_bag_crc32_conv = conv.encode_conv(send_bag_crc32)  # 对crc32编码后的数据包进行卷积编码
        send_data.append(send_bag_crc32_conv)  # 将编码后的数据包添加到发送队列中
    send_bag = img_compress[(send_data_length - 1) * packet_size * 8:]  # 取出最后一个待发送的数据包
    send_bag_crc32 = crc32(send_bag)  # 对待发送的数据包进行crc32编码
    send_bag_crc32_conv = conv.encode_conv(send_bag_crc32)  # 对crc32编码后的数据包进行卷积编码
    send_data.append(send_bag_crc32_conv)  # 将编码后的数据包添加到发送队列中
    for packet in send_data:
        while True:
            try:
                # 将二进制位列表转换为字节串
                packet_bytes = get_bytes_from_binaryarray(packet)
                # 发送数据包
                sock.sendall(struct.pack('!I', packet_bytes.__len__())+packet_bytes)
                # 设置超时时间
                sock.settimeout(timeout)
                # 等待 ACK
                ack = sock.recv(2)
                if ack == b'OK':
                    print("Packet sent successfully and ACK received")
                    break
                elif ack == b'NO':
                    print("Packet sent successfully but ACK not received")
            except socket.timeout:
                print("No ACK received, resending packet...")
            except Exception as e:
                print(f"An error occurred: {e}")
                break

# 主函数
def main():
    k = 3  # 卷积编码器约束长度
    packet_size = 1024 # 数据包大小（字节）
    start_sender('./images/image.bmp', k, packet_size)

if __name__ == "__main__":
    main()