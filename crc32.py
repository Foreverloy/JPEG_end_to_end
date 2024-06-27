def crc32(data: list, poly=0xEDB88320, init=0xFFFFFFFF):
    """
    计算给定数据的CRC32值，并将其附加到数据后面。

    :param data: 要计算CRC的数据，以二进制位列表形式传入。
    :param poly: CRC多项式，默认为0xEDB88320。
    :param init: CRC初始值，默认为0xFFFFFFFF。
    :return: 原始数据和附加的CRC32值，以二进制数列形式返回。
    """
    # 将二进制位列表转换为字节串
    data_bytes = int(''.join(str(b) for b in data), 2).to_bytes((len(data) + 7) // 8, byteorder='big')
    crc = init
    for byte in data_bytes:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
    
    # 计算得到的CRC32值
    crc_value = [int(bit) for bit in bin(crc ^ 0xFFFFFFFF)[2:].zfill(32)]  # 确保返回的二进制数列是32位
    
    # 返回原始数据和附加的CRC32值的组合
    return data + crc_value

def verify_crc32(data_with_crc: list, poly=0xEDB88320, init=0xFFFFFFFF):
    """
    验证给定数据（包括附加的CRC32值）的完整性。
    :param data_with_crc: 包含数据和附加CRC32值的二进制位列表。
    :param poly: CRC多项式，默认为0xEDB88320。
    :param init: CRC初始值，默认为0xFFFFFFFF。
    :return: 如果计算得到的CRC32值与附加的CRC32值匹配，则返回True；否则返回False。
    """
    data_length = len(data_with_crc) - 32
    data = data_with_crc[:data_length]
    received_crc = data_with_crc[data_length:]
    calculated_crc = crc32(data, poly, init)[-32:]  # 计算数据部分的CRC32值并提取最后32位
    return calculated_crc == received_crc