import numpy as np


class Convolution(object):
    # 实现（2，1，k）卷积编码，k为约束长度，取值为2-8
    k=5
    gen_polynomials = {
        2: ([1, 1, 1], [1, 0, 1]),  # k = 2
        3: ([1, 1, 0, 1], [1, 1, 1, 1]),  # k = 3
        4: ([1, 1, 1, 0, 1], [1, 0, 1, 1, 1]),  # k = 4
        5: ([1, 1, 1, 1, 0, 1], [1, 0, 1, 0, 1, 1]),  # k = 5
        6: ([1, 1, 1, 1, 0, 0, 1], [1, 0, 1, 1, 0, 1, 1]),  # k = 6
        7: ([1, 1, 1, 1, 1, 0, 1, 1], [1, 0, 1,0, 1, 1, 0, 1]),  # k = 7
        8: ([1, 1, 1, 1, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 1, 0, 1, 1])  # k = 8
    }
    def __init__(self,k:int):
        if not 2 <= k <= 8:
            raise ValueError("k must be between 2 and 8")
        self.k=k
    def encode_conv(self,x):  #k为卷积码的约束长度
        # x = []
        # # 将16进制字符串转换为二进制列表，用于卷积编码
        # for i in image_data:
        #     i = int(i, 16)
        #     num = []
        #     for j in range(4):
        #         num.append(i % 2)
        #         i = i // 2
        #     num.reverse()
        #     x.extend(num)
        # 存储编码信息
        y = []
        # k个寄存器初始化为0
        registers = [0] * self.k
        # 生成多项式
        G1 = self.gen_polynomials[self.k][0]  # 对应第一个输出比特
        G2 = self.gen_polynomials[self.k][1]  # 对应第二个输出比特
        for j in x:
            c_1 = sum([j * G1[0]] +
                      [registers[i] * G1[i + 1] for i in range(self.k)]) % 2
            c_2 = sum([j * G2[0]] +
                      [registers[i] * G2[i + 1] for i in range(self.k)]) % 2
            y.append(c_1)
            y.append(c_2)
            # 更新寄存器
            registers = [j] + registers[:-1]
        return y

    def decode_conv(self,y):
        # 生成多项式
        G1 = self.gen_polynomials[self.k][0]  # 对应第一个输出比特
        G2 = self.gen_polynomials[self.k][1]  # 对应第二个输出比特
        num_states = 2**self.k  # k个寄存器，共有2^k种状态
        # 初始化路径度量和路径跟踪
        score_list = np.full((num_states, len(y) // 2 + 1), np.inf)
        score_list[0][0] = 0  # 初始状态的路径度量为0
        # 记录回溯路径
        trace_back_list = []
        # 每个阶段的回溯块
        trace_block = []

        # 根据当前状态和输出求出编码后的输出
        def encode_with_state(x, state):
            registers = [(state >> i) & 1
                         for i in range(self.k-1, -1, -1)]  #将状态转化为二进制列表，存入寄存器
            # 编码后的输出
            y = []
            c_1 = sum([x * G1[0]] +
                      [registers[i] * G1[i + 1] for i in range(self.k)]) % 2
            c_2 = sum([x * G2[0]] +
                      [registers[i] * G2[i + 1] for i in range(self.k)]) % 2
            y.append(c_1)
            y.append(c_2)
            return y

        # 计算汉明距离
        def hamming_dist(y1, y2):
            dist = (y1[0] - y2[0]) % 2 + (y1[1] - y2[1]) % 2
            return dist

        # 根据当前状态now_state和输入信息比特input，算出下一个状态
        def state_transfer(input, now_state):
            return ((now_state >> 1) | (input << self.k-1)
                    )  #当前状态右移一位，输入左移6位，然后或运算，得到下一步的状态

        # 根据不同初始时刻更新参数
        # 也即指定状态为 state 时的参数更新
        # y_block 为 y 的一部分， shape=(2,)
        # pre_state 表示当前要处理的状态
        # index 指定需要处理的时刻
        def update_with_state(y_block, pre_state, index):
            # 输入的是 0
            encode_0 = encode_with_state(0, pre_state)  #得出当前输出
            next_state_0 = state_transfer(0, pre_state)  #得出下一个状态
            score_0 = hamming_dist(y_block,
                                   encode_0)  #计算由前一个状态和输入得到的输出和实际输出之间的汉明距离
            # 输入为0，且需要更新
            if score_list[pre_state][index] + score_0 < score_list[
                    next_state_0][index + 1]:
                score_list[next_state_0][
                    index + 1] = score_list[pre_state][index] + score_0
                trace_block[next_state_0][0] = pre_state
                trace_block[next_state_0][1] = 0
            # 输入的是 1
            encode_1 = encode_with_state(1, pre_state)
            next_state_1 = state_transfer(1, pre_state)
            score_1 = hamming_dist(y_block, encode_1)
            # 输入为1，且需要更新
            if score_list[pre_state][index] + score_1 < score_list[
                    next_state_1][index + 1]:  #完成汉明码的对比，得到最小的汉明距离
                score_list[next_state_1][
                    index + 1] = score_list[pre_state][index] + score_1
                trace_block[next_state_1][0] = pre_state
                trace_block[next_state_1][1] = 1
            if pre_state == num_states - 1 or index == 0:  #时刻已经结束，存储本时刻的状态信息
                trace_back_list.append(trace_block)

        # 默认寄存器初始为 00。也即，开始时刻，默认状态为00
        # 开始第一个 y_block 的更新
        y_block = y[0:2]
        trace_block = [[-1, -1] for _ in range(num_states)]  #初始化回溯块
        update_with_state(y_block=y_block, pre_state=0, index=0)
        # 开始之后的 y_block 更新
        for index in range(2, int(len(y)), 2):
            y_block = y[index:index + 2]
            for state in range(num_states):
                if state == 0:
                    trace_block = [[-1, -1]
                                   for _ in range(num_states)]  #初始化回溯块
                update_with_state(y_block=y_block,
                                  pre_state=state,
                                  index=int(index // 2))
        # 完成前向过程，开始回溯
        # state_trace_index 表示 开始回溯的状态是啥
        state_trace_index = np.argmin(
            score_list[:, -1]
        )  #拿到最后一列最小值在最后一列中的下标（行号），而在score_list中行号即为状态,即找到最后一个时刻的最小汉明距离对应 的状态
        # 记录原编码信息
        x = []
        for trace in range(len(trace_back_list) - 1, -1, -1):
            x.append(trace_back_list[trace][state_trace_index][1])
            state_trace_index = trace_back_list[trace][state_trace_index][0]
        x = list(reversed(x))
        #将解码得到的二进制列表转化为16进制字符串
        # image_data = ""  #压缩的图像数据
        # for i in range(len(x) // 4):
        #     tmp = ""
        #     for j in range(4):
        #         tmp += str(x[i * 4 + j])
        #     image_data += hex(int(tmp, 2))[2:]
        return x
