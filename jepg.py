import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
jpeg压缩函数
data:要压缩的灰度图像数据流
quality_scale控制压缩质量(1-99)，默认为50，值越小图像约清晰
return:得到压缩后的图像数据，为FFD9开头的jpeg格式字符串
'''


def compress(img_data, quality_scale=50):
    # 获取图像数据流宽高
    h, w ,_= img_data.shape
    # 标准亮度量化表
    Qy = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 58, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.uint8)

    # 根据压缩质量重新计算量化表
    if quality_scale <= 0:
        quality_scale = 1
    elif quality_scale >= 100:
        quality_scale = 99
    for i in range(64):
        tmp = int((Qy[int(i / 8)][i % 8] * quality_scale + 50) / 100)
        if tmp <= 0:
            tmp = 1
        elif tmp > 255:
            tmp = 255
        Qy[int(i / 8)][i % 8] = tmp

    # Z字型
    ZigZag = [
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63]

    # DC哈夫曼编码表
    standard_dc_nrcodes = [0, 0, 7, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    standard_dc_values = [4, 5, 3, 2, 6, 1, 0, 7, 8, 9, 10, 11]#默认直流部分哈夫曼表的顺序
    pos_in_table = 0;#standard_dc_nrcodes下标
    code_value = 0;#可以理解为standrad_dc_values的下标
    dc_huffman_table = [0] * 16

    for i in range(1, 9):
        for j in range(1, standard_dc_nrcodes[i - 1] + 1):
            dc_huffman_table[standard_dc_values[pos_in_table]] = bin(code_value)[2:].rjust(i, '0')#在huffman表的对应位置存储二进制编码，下标是对应的值，如下标4对应“000”、下标3对应“010“，哈夫曼编码的由对应数据在standard_dc_values中的下标得出，如4在该表中下标为0，3在该表中下标为2
            pos_in_table += 1
            code_value += 1
        code_value <<= 1

    # AC哈夫曼编码表
    standard_ac_nrcodes = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d]
    standard_ac_values = [0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
                          0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
                          0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
                          0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
                          0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
                          0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
                          0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
                          0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
                          0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
                          0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
                          0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
                          0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
                          0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
                          0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
                          0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
                          0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
                          0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
                          0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
                          0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
                          0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
                          0xf9, 0xfa]

    pos_in_table = 0;
    code_value = 0;
    ac_huffman_table = [0] * 256

    for i in range(1, 17):
        for j in range(1, standard_ac_nrcodes[i - 1] + 1):
            ac_huffman_table[standard_ac_values[pos_in_table]] = bin(code_value)[2:].rjust(i, '0')
            pos_in_table += 1
            code_value += 1
        code_value <<= 1
    # 转成float类型
    img_data = img_data.astype(np.float32)
    # 存储最后的哈夫曼编码
    result = ''
    prev = 0
    # 分成8*8的块
    for i in range(h // 8):
        for j in range(w // 8):
            block = img_data[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] - 128#图像的像素值通常在0到255之间，减去128后，像素值的范围变为-128到127。这样做的目的是为了让数据集中在0附近
            block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
            block = cv2.dct(block)
            # 数据量化
            block[:] = np.round(block / Qy)
            # 把量化后的二维矩阵转成一维数组
            arr = [0] * 64
            notnull_num = 0
            for k in range(64):
                tmp = int(block[int(k / 8)][k % 8])
                arr[ZigZag[k]] = tmp;
                # 统计arr数组中有多少个非0元素
                if tmp != 0:
                    notnull_num += 1
            # RLE编码
            # 标识连续0的个数
            time = 0
            # 处理DC
            if arr[0] != 0:
                notnull_num -= 1
            data = arr[0] - prev
            data2 = bin(np.abs(data))[2:]
            data1 = len(data2)
            if data < 0:
                data2 = bin(np.abs(data) ^ (2 ** data1 - 1))[2:].rjust(data1, '0')
            if data == 0:
                data2 = ''
                data1 = 0
            result += dc_huffman_table[data1]
            result += data2
            prev = arr[0]
            for k in range(1, 64):
                # 有可能AC系数全为0 所以要先进行判断
                if notnull_num == 0:
                    # 添加EOB
                    result += '1010'
                    break
                if arr[k] == 0 and time < 15:
                    time += 1
                else:
                    # BIT编码
                    # 处理括号中第二个数
                    data = arr[k]
                    data2 = bin(np.abs(data))[2:]
                    data1 = len(data2)
                    if data < 0:
                        data2 = bin(np.abs(data) ^ (2 ** data1 - 1))[2:].rjust(data1, '0')#np.abs(data)得到不带符号的数，异或运算得到原数的反码，再转成二进制
                    if data == 0:
                        data1 = 0
                        data2 = ''
                    # 哈夫曼编码，序列化
                    result += ac_huffman_table[time * 16 + data1]

                    result += data2
                    time = 0
                    # 判断是否要添加EOB
                    if int(arr[k]) != 0:
                        notnull_num -= 1
    # 补足为8的整数倍，以便编码成16进制数据
    if len(result) % 8 != 0:
        result = result.ljust(len(result) + 8 - len(result) % 8, '0')
    res_data = ''
    for i in range(0, len(result), 8):
        temp = int(result[i:i + 8], 2)
        res_data += hex(temp)[2:].rjust(2, '0').upper()
        if temp == 255:
            res_data += '00'
    result = res_data
    res = ''

    # 添加jpeg文件头
    # SOI(文件头),共89个字节
    res += 'FFD8'
    # APP0图像识别信息
    res += 'FFE000104A46494600010100000100010000'
    # DQT定义量化表
    res += 'FFDB004300'
    # 64字节的量化表

    for i in range(64):
        res += hex(Qy[int(i / 8)][i % 8])[2:].rjust(2, '0')
    # SOF0图像基本信息，13个字节
    res += 'FFC0000B08'
    res += hex(h)[2:].rjust(4, '0')
    res += hex(w)[2:].rjust(4, '0')
    # res+='01012200'
    # 采样系数好像都是1
    res += '01011100'
    # DHT定义huffman表,33个字节+183
    res += 'FFC4001F00'
    for i in standard_dc_nrcodes:
        res += hex(i)[2:].rjust(2, '0')
    for i in standard_dc_values:
        res += hex(i)[2:].rjust(2, '0')
    res += 'FFC400B510'
    for i in standard_ac_nrcodes:
        res += hex(i)[2:].rjust(2, '0')

    for i in standard_ac_values:
        res += hex(i)[2:].rjust(2, '0')

    # SOS扫描行开始，10个字节
    res += 'FFDA0008010100003F00'
    # 压缩的图像数据（一个个扫描行），数据存放顺序是从左到右、从上到下
    res += result
    # EOI文件尾0
    res += 'FFD9'
    return res


'''
jpeg解压缩
img:解压缩的jpeg灰度图像文件
return:返回解压缩后的图像原数据，为多维数组形式
'''


def decompress(img):
    # jpeg解码的所有参数都是从编码后的jpeg文件中读取的
    with open(img, 'rb') as f:
        img_data = f.read()
    res = ''
    for i in img_data:
        res += hex(i)[2:].rjust(2, '0').upper()

    ZigZag = [
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63]
    # 获取亮度量化表
    Qy = np.zeros((8, 8))
    for i in range(64):
        Qy[int(i / 8)][i % 8] = int(res[50 + i * 2:52 + i * 2], 16)
        # 获取SOF0图像基本信息，图像的宽高
    h = int(res[188:192], 16)
    w = int(res[192:196], 16)
    # 获取DHT定义huffman表
    standard_dc_values = res[246:270]
    standard_dc_nrcodes = [0] * 16
    for i in range(16):
        standard_dc_nrcodes[i] = int(res[214 + i * 2:216 + i * 2], 16)

    standard_ac_values = res[312:636]
    standard_ac_nrcodes = [0] * 16
    for i in range(16):
        standard_ac_nrcodes[i] = int(res[280 + i * 2:282 + i * 2], 16)
    # 生成dc哈夫曼表
    pos_in_table = 0;
    code_value = 0;
    reverse_dc_huffman_table = {}

    for i in range(1, 9):
        for j in range(1, standard_dc_nrcodes[i - 1] + 1):
            reverse_dc_huffman_table[bin(code_value)[2:].rjust(i, '0')] = standard_dc_values[
                                                                          pos_in_table * 2:pos_in_table * 2 + 2]
            pos_in_table += 1
            code_value += 1
        code_value <<= 1
        # 生成ac哈夫曼表
    pos_in_table = 0;
    code_value = 0;
    reverse_ac_huffman_table = {}

    for i in range(1, 17):
        for j in range(1, standard_ac_nrcodes[i - 1] + 1):
            reverse_ac_huffman_table[bin(code_value)[2:].rjust(i, '0')] = standard_ac_values[
                                                                          pos_in_table * 2:pos_in_table * 2 + 2]
            pos_in_table += 1
            code_value += 1
        code_value <<= 1

    # 获取压缩的图像数据
    tmp_result = res[656:-4]
    result = ''
    i = 0
    while i < len(tmp_result):
        tmp0 = tmp_result[i:i + 2]
        result += tmp0
        i += 2
        if (tmp0 == 'FF'):
            i += 2
    # 得到哈夫曼编码后的01字符串
    result = bin(int(result, 16))[2:].rjust(len(result) * 4, '0')

    img_data = np.zeros((h, w))
    pos = 0
    prev = 0
    for j in range(h // 8):
        for k in range(w // 8):
            # 逆dc哈夫曼编码
            # 正向最大匹配
            arr = [0]
            # 计算EOB块中0的个数
            num = 0
            for i in range(8, 2, -1):
                tmp = reverse_dc_huffman_table.get(result[pos:pos + i])
                # 匹配成功
                if (tmp):
                    pos += i
                    num += 1
                    # DC系数为0
                    if tmp == '00':
                        # 是差值编码 差点忘了加上prev
                        arr[0] = 0 + prev
                        prev = arr[0]
                        break
                    data1 = int(tmp[1], 16)
                    data2 = result[pos:pos + data1]
                    if data2[0] == '0':
                        # 负数
                        data = -(int(data2, 2) ^ (2 ** data1 - 1))
                    else:
                        data = int(data2, 2)
                    arr[0] = data + prev
                    prev = arr[0]
                    pos += data1
                    break
            # 逆ac哈夫曼编码
            while (num < 64):
                # AC系数编码长度是从16bits到2bits
                for i in range(16, 1, -1):
                    tmp = reverse_ac_huffman_table.get(result[pos:pos + i])
                    if (tmp):
                        pos += i
                        if (tmp == '00'):
                            arr += ([0] * (64 - num))
                            num = 64
                            break
                        time = int(tmp[0], 16)
                        data1 = int(tmp[1], 16)
                        data2 = result[pos:pos + data1]
                        pos += data1
                        # data2为空，赋值为0，应对(15,0)这种情况
                        data2 = data2 if data2 else '0'
                        if data2[0] == '0':
                            # 负数,注意负号和异或运算的优先级
                            data = -(int(data2, 2) ^ (2 ** data1 - 1))
                        else:
                            data = int(data2, 2)
                        num += time + 1
                        # time个0
                        arr += ([0] * time)
                        # 非零值或最后一个单元0
                        arr.append(data)
                        break
            # 逆ZigZag扫描,得到block量化块
            block = np.zeros((8, 8))
            for i in range(64):
                block[int(i / 8)][i % 8] = arr[ZigZag[i]]
            # 逆量化
            block = block * Qy
            # 逆DCT变换
            block = cv2.idct(block)
            img_data[j * 8:(j + 1) * 8, k * 8:(k + 1) * 8] = block + 128
    img_data = img_data.astype(np.uint8)
    return img_data
