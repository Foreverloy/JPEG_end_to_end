import socket
from crc32 import verify_crc32, crc32
from convolutional_coding import Convolution
from jpeg import compress, decompress
from hexString_to_bitList import get_bitList_from_hexString, get_hexString_from_bitList, get_binaryarray_from_bytes
from awgn import add_awgn_noise
import base64
import matplotlib.pyplot as plt
import numpy as np
import struct
import cv2

def recv_message(sock):
    # 获取每个数据包头部存储数据包长度的四个字节
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('!I', raw_msglen)[0] #将获得的数据包长度转换为整数
    # 根据数据包长度接收数据包
    return recvall(sock, msglen)

def recvall(sock, n):
    # 从套接字接收n字节数据
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data
# 接收端函数
def start_receiver(k: int, packet_size: int):
    """
    接收端函数,在接收端完成数据包的解码，卷积解码，CRC校验，数据包的重组
    k: 卷积编码器约束长度
    packet_size: 数据包大小
    return: 接收到的数据流（二进制列表）
    """
    print("Receiver is running...")
    # 实例化一个(2,1,k)卷积编码器，2 <= k <= 8
    conv = Convolution(k)
    # 创建 TCP 套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', 10000)
    sock.bind(server_address)
    sock.listen(1)
    print("Receiver is listening on port 10000")
    while True:
        connection, client_address = sock.accept()
        recv_data = []  # 接收到的数据流
        fault_packet = 0  # 错误数据包数量
        try:
            while True:
                data_bag = recv_message(connection)
                if data_bag:
                    data_bag = get_binaryarray_from_bytes(
                        data_bag)  # 字节串转换为二进制列表
                    data_bag=add_awgn_noise(data_bag, 10)  # 添加高斯白噪声
                    data_bag = conv.decode_conv(data_bag)  # 卷积解码
                    if verify_crc32(data_bag):
                        print("Data is valid")
                        recv_data.extend(data_bag[:-32])
                        # 发送 ACK
                        connection.sendall(b'OK')
                    else:
                        print("Data is invalid")
                        fault_packet += 1
                        # 发送 NAK
                        connection.sendall(b'NO')
                else:
                    if recv_data:#如果data_bag为空，且recv_data不为空，说明数据接收完成
                        print("Data received successfully")
                        break
        finally:
            connection.close()
        # 数据接收完成后处理数据
        if recv_data:
            print("Fault packet number: ", fault_packet)
            connection.close()
            return recv_data
def main():
    k = 3 # 卷积编码器约束长度
    packet_size = 1024 # 数据包大小（字节）
    recv_data=start_receiver(k, packet_size)
    # 将接收到的数据流转换为16进制字符串
    recv_data = get_hexString_from_bitList(recv_data)
    # 存储压缩后的图像
    img_compress_path = './receive_image/img_compress.jpg'
    with open(img_compress_path, 'wb') as f:
        f.write(base64.b16decode(recv_data.upper()))
    # jpeg解压缩
    img_decompress = decompress(img_compress_path)
    # 原始图像路径,灰度图像
    img_path = './sender_image/image.png'
    # 读取原始图像,cv2.imread()默认是用color模式读取的，保持原样读取要加上第二个参数-1,即CV_LOAD_IMAGE_GRAYSCALE
    # 得到图像原数据流
    img_data = cv2.imread(img_path, -1)[:,:,(2,1,0)]
    # 官方jpeg压缩
    cv2.imwrite('./sender_image/jpeg_decompress.jpg', img_data)
    # 官方解压缩
    img0 = cv2.imread('./sender_image/jpeg_decompress.jpg', -1)
    # 读取接收端接收的压缩的图像
    img1 = cv2.imread('./receive_image/img_compress.jpg', -1)
    # 结果展示
    # 子图1，原始图像
    plt.subplot(141)
    # imshow()对图像进行处理，画出图像，show()进行图像显示
    plt.imshow(img_data, cmap=plt.cm.gray)
    plt.title('Oringinal Image')
    # 不显示坐标轴
    plt.axis('off')

    # 子图2，自己写的jpeg压缩后解压的图像
    plt.subplot(143)
    plt.imshow(img_decompress, cmap=plt.cm.gray)
    plt.title('my decompressed jpeg image')
    plt.axis('off')

    # 子图3，官方jpeg压缩后解码图像
    plt.subplot(144)
    plt.imshow(img0, cmap=plt.cm.gray)
    plt.title('formal decompressed jpeg image')
    plt.axis('off')

    # 子图4，jpeg压缩后图像
    plt.subplot(142)
    plt.imshow(img1, cmap=plt.cm.gray)
    plt.title('jpeg compressed image')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
