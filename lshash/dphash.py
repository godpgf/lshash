import numpy as np


class DPHash(object):
    def __init__(self):
        self.index_list = []
        self.extra_list = []
        self.hash_table = []

    def index(self, input_point, extra_data):
        if len(self.index_list) > 0:
            assert len(input_point) == len(self.index_list[-1])
        self.index_list.append(input_point)
        self.extra_list.append(extra_data)

    def create_hash_table(self):
        for i in range(len(self.index_list[0])):
            data_array = np.empty(len(self.index_list))
            for j in range(len(self.index_list)):
                data_array[j] = self.index_list[j][i]
            self.hash_table.append(np.argsort(data_array))

    def query(self, query_point, num_results):
        assert len(query_point) == len(self.hash_table) and len(query_point) == len(self.index_list[0])
        ids_set = set()
        sub_dot_list = []
        for i in range(len(query_point)):
            if query_point[i] >= 0:
                d0 = -1
            else:
                d0 = 0
            id = self.hash_table[i][d0]
            value = query_point[i] * self.index_list[id][i]
            sub_dot_list.append((i, d0, value))
        sub_dot_list.sort(key=lambda x: x[2])
        # 记录第i维，排名第d0个元素，子点积value，按照子点积从小到大排序

        while len(ids_set) < num_results and len(sub_dot_list) > 0:
            i, d0 = sub_dot_list[-1][0], sub_dot_list[-1][1]
            id = self.hash_table[i][d0]
            ids_set.add(id)
            id = -1
            if d0 >= 0:
                d0 += 1
                if d0 < len(self.index_list):
                    id = self.hash_table[i][d0]
            else:
                d0 -= 1
                if d0 >= -len(self.index_list):
                    id = self.hash_table[i][d0]

            if id >= 0:
                value = query_point[i] * self.index_list[id][i]
                j = len(sub_dot_list) - 2
                while j >= -1:
                    if j >= 0 and sub_dot_list[j][2] > value:
                        sub_dot_list[j+1] = sub_dot_list[j]
                    else:
                        sub_dot_list[j+1] = (i, id, value)
                        break
                    j -= 1
            else:
                sub_dot_list = sub_dot_list[:-1]
        return [self.extra_list[id] for id in ids_set]
