def crc32(data:list, poly=0xEDB88320, init=0xFFFFFFFF):
    """
    计算给定数据的CRC32值。

    :param data: 要计算CRC的数据，以字节数组形式传入。
    :param poly: CRC多项式，默认为0xEDB88320。
    :param init: CRC初始值，默认为0xFFFFFFFF。
    :return: 计算得到的CRC32值。
    """
    data_bytes =  ''.join(str(b) for b in data)
    data_bytes = data_bytes.encode()
    crc = init
    for byte in data_bytes:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF

def verify_crc32(data:list, expected_crc: int, poly=0xEDB88320, init=0xFFFFFFFF):
    """
    验证给定数据的CRC32值是否正确。

    :param data: 要验证CRC的数据，以字节数组形式传入。
    :param expected_crc: 预期的CRC32值。
    :param poly: CRC多项式，默认为0xEDB88320。
    :param init: CRC初始值，默认为0xFFFFFFFF。
    :return: 如果计算得到的CRC32值与预期值匹配，则返回True；否则返回False。
    """
    calculated_crc = crc32(data, poly, init)
    return calculated_crc == expected_crc

# 示例数据
data = b"123456789"

# 计算CRC32
crc_result = crc32(data)
print(f"CRC32结果: {crc_result:08X}")

# 验证CRC32
data = b"123556789"
is_valid = verify_crc32(data, crc_result)
print(f"CRC32校验结果: {'通过' if is_valid else '失败'}")