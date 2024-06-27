import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64

'''
rgb到ycbcr颜色空间转换
'''

#RGB -> YCbCr 
def rgb2ycbcr(rgb):
    m = np.array([[0.299, 0.587, 0.114],
                  [-0.1687, -0.3313, 0.5],
                  [0.5, -0.4187, -0.0813]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose())
    return ycbcr.reshape(shape)


# YCbCr -> RGB
def ycbcr2rgb(ycbcr):
    m = np.array([[1, 0, 1.402],
                  [1, -0.344, -0.714],
                  [1, 1.772, 0]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = np.dot(ycbcr, m.transpose())
    return rgb.reshape(shape)

def _computeHuffmanTable(nr_codes,std_table,huffman_table):
	pos_in_table=0;
	code_value=0
	for k in range(1,17):
		for j in range(1,nr_codes[k-1]+1):
			huffman_table[std_table[pos_in_table]]=bin(code_value)[2:].rjust(k,'0')
			pos_in_table+=1
			code_value+=1
		code_value<<=1

#传入8*8块
def _doHuffmanEncoding(block,ZigZag,m_Table,m_DC_Huffman_Table,m_AC_Huffman_Table,result,prev):
    block=block.astype(np.float64)
    block = cv2.dct(block)
    # 数据量化
    block[:] = np.round(block / m_Table)
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
    data = arr[0] - prev[0]
    prev[0] = arr[0]
    data2 = bin(np.abs(data))[2:]
    data1 = len(data2)
    if data < 0:
        data2 = bin(np.abs(data) ^ (2 ** data1 - 1))[2:].rjust(data1, '0')
    if data == 0:
        data2 = ''
        data1 = 0
    result += m_DC_Huffman_Table[data1]
    result += data2

    for k in range(1, 64):
        # 有可能AC系数全为0 所以要先进行判断
        if notnull_num == 0:
            # 添加EOB
            result += m_AC_Huffman_Table[0]
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
                data2 = bin(np.abs(data) ^ (2 ** data1 - 1))[2:].rjust(data1, '0')
            if data == 0:
                data1 = 0
                data2 = ''
            # 哈夫曼编码，序列化
            result += m_AC_Huffman_Table[time * 16 + data1]
            result += data2
            time = 0
            # 判断是否要添加EOB
            if int(arr[k]) != 0:
                notnull_num -= 1
    return result
def _computeReverseHuffmanTable(nr_codes,nr_values,reverse_huffman_table):
    pos_in_table = 0;
    code_value = 0
    for i in range(1, 17):
        for j in range(1, nr_codes[i - 1] + 1):
            reverse_huffman_table[bin(code_value)[2:].rjust(i, '0')] = nr_values[pos_in_table * 2:pos_in_table * 2 + 2]
            pos_in_table += 1
            code_value += 1
        code_value <<= 1


'''
jpeg压缩函数
data:要压缩的彩色图像数据流
quality_scale控制压缩质量(1-99)，默认为50，值越小图像约清晰
return:得到压缩后的图像数据，为FFD9开头的jpeg格式字符串
'''

def compress(img_data, quality_scale=50):
    # 获取图像数据流宽高
    m_height, m_width,_= img_data.shape
    m_YTable = np.zeros([8,8], dtype=np.uint8)
    m_CbCrTable = np.zeros([8,8], dtype=int)

    # 标准亮度量化表
    Luminance_Quantization_Table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 58, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.uint8)

    # 标准色度量化表
    Chrominance_Quantization_Table = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                   [18, 21, 26, 66, 99, 99, 99, 99],
                   [24, 26, 56, 99, 99, 99, 99, 99],
                   [47, 66, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99]], dtype=np.uint8)
    # Z字型
    ZigZag = np.array([
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63])

    Standard_DC_Luminance_NRCodes = [0, 0, 7, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    Standard_DC_Luminance_Values = [4, 5, 3, 2, 6, 1, 0, 7, 8, 9, 10, 11]

    Standard_DC_Chrominance_NRCodes = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    Standard_DC_Chrominance_Values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    Standard_AC_Luminance_NRCodes = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d]
    Standard_AC_Luminance_Values = [0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
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

    Standard_AC_Chrominance_NRCodes = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77]

    Standard_AC_Chrominance_Values = [0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
                                      0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
                                      0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
                                      0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
                                      0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
                                      0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
                                      0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
                                      0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
                                      0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
                                      0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
                                      0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
                                      0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
                                      0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
                                      0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
                                      0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
                                      0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
                                      0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
                                      0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
                                      0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
                                      0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
                                      0xf9, 0xfa]

    #初始化量化表，根据压缩质量重新计算量化表
    if quality_scale <= 0:
        quality_scale = 1
    elif quality_scale >= 100:
        quality_scale = 99
    for i in range(64):
        # 亮度量化表
        tmp = int((Luminance_Quantization_Table[int(i / 8)][i % 8] * quality_scale + 50) / 100)
        if tmp <= 0:
            tmp = 1
        elif tmp > 255:
            tmp = 255
        m_YTable[int(i / 8)][i % 8] = tmp

        # 色度量化表
        tmp = int((Chrominance_Quantization_Table[int(i / 8)][i % 8] * quality_scale + 50) / 100)
        if tmp <= 0:
            tmp = 1
        elif tmp > 255:
            tmp = 255
        m_CbCrTable[int(i / 8)][i % 8] = tmp

    # 初始化哈夫曼编码表
    m_Y_DC_Huffman_Table = [0]*12
    m_Y_AC_Huffman_Table = [0]*256
    m_CbCr_DC_Huffman_Table = [0]*12
    m_CbCr_AC_Huffman_Table = [0]*256
    _computeHuffmanTable(Standard_DC_Luminance_NRCodes, Standard_DC_Luminance_Values, m_Y_DC_Huffman_Table);
    _computeHuffmanTable(Standard_AC_Luminance_NRCodes, Standard_AC_Luminance_Values, m_Y_AC_Huffman_Table);
    _computeHuffmanTable(Standard_DC_Chrominance_NRCodes, Standard_DC_Chrominance_Values, m_CbCr_DC_Huffman_Table);
    _computeHuffmanTable(Standard_AC_Chrominance_NRCodes, Standard_AC_Chrominance_Values, m_CbCr_AC_Huffman_Table);

    # 转成float类型
    img_data = img_data.astype(np.float64)

    # 存储最后的哈夫曼编码
    result = ''

    #色彩空间转换
    YCbCr_data=rgb2ycbcr(img_data)
    YCbCr_data=YCbCr_data.astype(int)
    Y_data, Cb_data, Cr_data = cv2.split(YCbCr_data)
    Y_data=Y_data-128
    prev_DC_Y = [0]
    prev_DC_Cb = [0]
    prev_DC_Cr = [0]

    #CbCr降采样
    # Cb_data=Cb_data[::2,::2]
    # Cr_data=Cb_data[1::2,::2]
    #三个通道分别编码成独立数据流
    h,w=Y_data.shape
    # 分成8*8的块
    for i in range(h // 8):
        for j in range(w // 8):
            block = Y_data[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            result=_doHuffmanEncoding(block,ZigZag,m_YTable,m_Y_DC_Huffman_Table,m_Y_AC_Huffman_Table,result,prev_DC_Y)
            block = Cb_data[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            result=_doHuffmanEncoding(block,ZigZag,m_CbCrTable,m_CbCr_DC_Huffman_Table,m_CbCr_AC_Huffman_Table,result,prev_DC_Cb)
            block = Cr_data[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8]
            result=_doHuffmanEncoding(block,ZigZag,m_CbCrTable,m_CbCr_DC_Huffman_Table,m_CbCr_AC_Huffman_Table,result,prev_DC_Cr)


    # 补足为8的整数倍，以便编码成16进制数据
    if len(result) % 8 != 0:
        result = result.ljust(len(result) + 8 - len(result) % 8, '0')
    res_data = ''
    for i in range(0, len(result), 8):
        temp = int(result[i:i + 8], 2)
        res_data += hex(temp)[2:].rjust(2, '0').upper()
        #如果16进制对应的字符是FF的话需要添加00辅助字节来区分（FF是JPEG段标识）
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
    res += 'FFDB008400'
    # 64字节的量化表
    for i in range(64):
        res += hex(m_YTable[int(i / 8)][i % 8])[2:].rjust(2, '0')
    res+='01'
    for i in range(64):
        res += hex(m_CbCrTable[int(i / 8)][i % 8])[2:].rjust(2, '0')
        
    # SOF0图像基本信息，13个字节
    res += 'FFC0001108'
    res += hex(m_height)[2:].rjust(4, '0')
    res += hex(m_width)[2:].rjust(4, '0')
    #降采样4:2:0
    # res += '03011100022201032201'
    #全采样
    res += '03011100021101031101'
    # DHT定义huffman表,33个字节+183
    res += 'FFC401A200'
    for i in Standard_DC_Luminance_NRCodes:
        res += hex(i)[2:].rjust(2, '0')
    for i in Standard_DC_Luminance_Values:
        res += hex(i)[2:].rjust(2, '0')
    res += '10'
    for i in Standard_AC_Luminance_NRCodes:
        res += hex(i)[2:].rjust(2, '0')
    for i in Standard_AC_Luminance_Values:
        res += hex(i)[2:].rjust(2, '0')
    res += '01'
    for i in Standard_DC_Chrominance_NRCodes:
        res += hex(i)[2:].rjust(2, '0')
    for i in Standard_DC_Chrominance_Values:
        res += hex(i)[2:].rjust(2, '0')
    res += '11'
    for i in Standard_AC_Chrominance_NRCodes:
        res += hex(i)[2:].rjust(2, '0')
    for i in Standard_AC_Chrominance_Values:
        res += hex(i)[2:].rjust(2, '0')
    #SOS扫描行开始，10个字节
    res += 'FFDA000C03010002110311003F00'
    # 压缩的图像数据（一个个扫描行），数据存放顺序是从左到右、从上到下
    res += result
    # EOI文件尾0
    res += 'FFD9'
    return res

def _doHuffmanDecoding(m_Table,ZigZag,Reverse_DC_Huffman_Table,Reverse_AC_Huffman_Table,result,prev,pos,img_data,j,k):
    # 逆dc哈夫曼编码
    # 正向最大匹配
    arr = [0]
    # 计算EOB块中0的个数
    num = 0
    #这里注意CbCr的哈夫曼表的范围
    for i in range(11, 1, -1):
        tmp = Reverse_DC_Huffman_Table.get(result[pos[0]:pos[0] + i])
        # 匹配成功
        if (tmp):
            pos[0] += i
            num += 1
            # DC系数为0
            if tmp == '00':
                # 是差值编码 差点忘了加上prev
                arr[0] = 0 + prev[0]
                prev[0] = arr[0]
                break

            time = 0
            data1 = int(tmp[1], 16)
            data2 = result[pos[0]:pos[0] + data1]
            if data2[0] == '0':
                # 负数
                data = -(int(data2, 2) ^ (2 ** data1 - 1))
            else:
                data = int(data2, 2)
            arr[0] = data + prev[0]
            prev[0] = arr[0]
            pos[0] += data1
            break
    # 逆ac哈夫曼编码
    while (num < 64):
        # AC系数编码长度是从16bits到2bits
        for i in range(16, 1, -1):
            tmp = Reverse_AC_Huffman_Table.get(result[pos[0]:pos[0] + i])
            if (tmp):
                pos[0] += i
                if (tmp == '00'):
                    arr += ([0] * (64 - num))
                    num = 64
                    break
                time = int(tmp[0], 16)
                data1 = int(tmp[1], 16)
                data2 = result[pos[0]:pos[0] + data1]
                pos[0] += data1
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
    block = block * m_Table

    # 逆DCT变换
    block = cv2.idct(block)
    img_data[j * 8:(j + 1) * 8, k * 8:(k + 1) * 8] = block


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
    m_YTable = np.zeros((8, 8))
    for i in range(64):
        m_YTable[int(i / 8)][i % 8] = int(res[50 + i * 2:52 + i * 2], 16)
    m_CbCrTable = np.zeros((8, 8))
    for i in range(64):
        m_CbCrTable[int(i / 8)][i % 8] = int(res[180 + i * 2:182 + i * 2], 16)

    # 获取SOF0图像基本信息，图像的宽高
    h = int(res[318:322], 16)
    w = int(res[322:326], 16)
    
    # 获取DHT定义huffman表
    Standard_DC_Luminance_NRCodes = [0] * 16
    for i in range(16):
        Standard_DC_Luminance_NRCodes[i] = int(res[356 + i * 2:358 + i * 2], 16)
    Standard_DC_Luminance_Values = res[388:412]

    Standard_AC_Luminance_NRCodes = [0] * 16
    for i in range(16):
        Standard_AC_Luminance_NRCodes[i] = int(res[414 + i * 2:416 + i * 2], 16)
    Standard_AC_Luminance_Values = res[446:770]

    Standard_DC_Chrominance_NRCodes=[0]*16
    for i in range(16):
        Standard_DC_Chrominance_NRCodes[i] = int(res[772 + i * 2:774 + i * 2], 16)
    Standard_DC_Chrominance_Values=res[804:828]

    Standard_AC_Chrominance_NRCodes=[0] * 16
    for i in range(16):
        Standard_AC_Chrominance_NRCodes[i] = int(res[830 + i * 2:832 + i * 2], 16)
    Standard_AC_Chrominance_Values=res[862:1186]

    #生成逆huffman编码表
    Reverse_Y_DC_Huffman_Table = {}
    Reverse_Y_AC_Huffman_Table = {}
    Reverse_CbCr_DC_Huffman_Table = {}
    Reverse_CbCr_AC_Huffman_Table = {}
    _computeReverseHuffmanTable(Standard_DC_Luminance_NRCodes, Standard_DC_Luminance_Values, Reverse_Y_DC_Huffman_Table);
    _computeReverseHuffmanTable(Standard_AC_Luminance_NRCodes, Standard_AC_Luminance_Values, Reverse_Y_AC_Huffman_Table);
    _computeReverseHuffmanTable(Standard_DC_Chrominance_NRCodes, Standard_DC_Chrominance_Values, Reverse_CbCr_DC_Huffman_Table);
    _computeReverseHuffmanTable(Standard_AC_Chrominance_NRCodes, Standard_AC_Chrominance_Values, Reverse_CbCr_AC_Huffman_Table);

    # 获取压缩的图像数据
    tmp_result = res[1214:-4]
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
    prev_DC_Y = [0]
    prev_DC_Cb = [0]
    prev_DC_Cr = [0]
    pos = [0]
    #逆huffman编码
    Y_data = np.zeros((h, w),dtype=int)
    Cb_data = np.zeros((h, w),dtype=int)
    Cr_data = np.zeros((h, w),dtype=int)
    for j in range(h // 8):
        for k in range(w // 8):
            _doHuffmanDecoding(m_YTable,ZigZag,Reverse_Y_DC_Huffman_Table,Reverse_Y_AC_Huffman_Table,result,prev_DC_Y,pos,Y_data,j,k)
            _doHuffmanDecoding(m_CbCrTable,ZigZag,Reverse_CbCr_DC_Huffman_Table,Reverse_CbCr_AC_Huffman_Table,result,prev_DC_Cb,pos,Cb_data,j,k)
            _doHuffmanDecoding(m_CbCrTable,ZigZag,Reverse_CbCr_DC_Huffman_Table,Reverse_CbCr_AC_Huffman_Table,result,prev_DC_Cr,pos,Cr_data,j,k)
    print(Y_data)
    Y_data=Y_data+128
    YCbCr_data=cv2.merge([Y_data,Cb_data,Cr_data])
    img_data=ycbcr2rgb(YCbCr_data)
    img_data = img_data.astype(np.uint8)
    return img_data


#huffman表需要修改
def main():
    # 原始图像路径,彩色图像
    img_path = './sender_image/image.png'
    # 读取原始图像
    # 得到图像原数据流，注意opencv的颜色通道顺序为[B,G,R]
    img_data = cv2.imread(img_path)[:,:,(2,1,0)]
    #直接把原始图像存储起来，得到官方压缩的jpeg图像数据img0
    cv2.imwrite('./jpeg_compress.jpg', img_data)
    img0 = cv2.imread('./jpeg_compress.jpg', -1)
    # 本文代码得到压缩后图像数据
    img_compress = compress(img_data, 50)
    # 存储压缩后的图像
    img_compress_path = './img_compress.jpg'
    with open(img_compress_path, 'wb') as f:
        f.write(base64.b16decode(img_compress.upper()))
    # jpeg图像解压缩测试
    img_decompress = decompress(img_compress_path)
    img1 = cv2.imread(img_compress_path)[:,:,(2,1,0)]

    # 结果展示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
    # 子图1，原始图像
    plt.subplot(221)
    # imshow()对图像进行处理，画出图像，show()进行图像显示
    plt.imshow(img_data)
    plt.title('原始图像')
    # 不显示坐标轴
    plt.axis('off')
    # 子图2，自己写的jpeg压缩后解码图像
    plt.subplot(222)
    plt.imshow(img_decompress, cmap=plt.cm.gray)
    plt.title('自写编码自写解码')
    plt.axis('off')

    # 子图3，jpeg压缩后解码图像
    plt.subplot(223)
    plt.imshow(img0, cmap=plt.cm.gray)
    plt.title('官方编码官方解码')
    plt.axis('off')

    # 子图3，jpeg压缩后解码图像
    plt.subplot(224)
    plt.imshow(img1, cmap=plt.cm.gray)
    plt.title('自写编码官方解码)')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()

