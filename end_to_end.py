from convolutional_coding import encode_conv,decode_conv
from jepg import compress,decompress
from awgn import add_awgn_noise
import matplotlib.pyplot as plt
import cv2
import base64
PATH='./images/image.bmp'

def main():
    # 原始图像路径,灰度图像
    img_path = './images/image.bmp'
    # 读取原始图像,cv2.imread()默认是用color模式读取的，保持原样读取要加上第二个参数-1,即CV_LOAD_IMAGE_GRAYSCALE
    # 得到图像原数据流
    img_data = cv2.imread(img_path, -1)
    cv2.imwrite('./images/jpeg_decompress.jpg', img_data)
    img0 = cv2.imread('./images/jpeg_decompress.jpg', -1)
    # 得到压缩后图像数据
    img_compress = compress(img_data, 50)
    # 对压缩后的数据进行卷积编码
    img_compress = encode_conv(img_compress)
    # 添加高斯白噪声，模拟通过awgn信道
    img_compress=add_awgn_noise(img_compress,10)
    # 对接收到的数据进行viterbi解码
    img_compress = decode_conv(img_compress)
    print("encode and decode successfully")
    # 存储压缩后的图像
    img_compress_path = './images/img_compress.jpg'
    with open(img_compress_path, 'wb') as f:
        f.write(base64.b16decode(img_compress.upper()))
    # jpeg图像解压缩测试
    img_decompress = decompress(img_compress_path)
    img1 = cv2.imread('./images/img_compress.jpg', -1)
    
    # 结果展示
    # 子图1，原始图像
    plt.subplot(141)
    # imshow()对图像进行处理，画出图像，show()进行图像显示
    plt.imshow(img_data, cmap=plt.cm.gray)
    plt.title('Oringinal Image')
    # 不显示坐标轴
    plt.axis('off')

    # 子图2，自己写的jpeg压缩后解码图像
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




if __name__ == '__main__':
    main()
