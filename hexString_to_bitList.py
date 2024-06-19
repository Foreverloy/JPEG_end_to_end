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

def get_binaryarray_from_bytes(bytes_data):
    bit_list = []
    for byte in bytes_data:
        # Convert byte to 8-bit binary string
        bin_str = format(byte, '08b')
        # Extend the list with individual bits
        bit_list.extend(int(bit) for bit in bin_str)
    return bit_list

def get_bytes_from_binaryarray(binary_array):
    # 确保二进制列表长度是8的倍数
    assert len(binary_array) % 8 == 0, "Binary array length must be a multiple of 8"
    
    bytes_data = bytearray()
    for i in range(0, len(binary_array), 8):
        # 取出8个二进制位的子列表
        byte_bits = binary_array[i:i+8]
        # 将二进制位列表转换为字符串
        byte_str = ''.join(str(bit) for bit in byte_bits)
        # 将二进制字符串转换为整数，然后转换为字节
        byte = int(byte_str, 2)
        bytes_data.append(byte)
    
    return bytes(bytes_data)