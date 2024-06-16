def get_bitList_from_hexString(hexString):
    x = []
    # 将16进制字符串转换为二进制列表，用于卷积编码
    for i in hexString:
        i = int(i, 16)
        num = []
        for j in range(4):
            num.append(i % 2)
            i = i // 2
        num.reverse()
        x.extend(num)
    return x


def get_hexString_from_bitList(bitlist):
    #将二进制列表转化为16进制字符串
    hexstring = ""  #压缩的图像数据
    for i in range(len(bitlist) // 4):
        tmp = ""
        for j in range(4):
            tmp += str(bitlist[i * 4 + j])
        hexstring += hex(int(tmp, 2))[2:]
    return hexstring