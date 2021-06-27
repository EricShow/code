from typing import List


class solution:

    def __init__(self, weight_max: int, weight_ls: List[int], value_ls: List[int]):
        self.weight_max = weight_max
        self.weight_ls = weight_ls
        self.value_ls = value_ls
        self.path = []
        self.maxvalue = 0

    def maxValue(self) -> int:
        self.dfs(0, 0, 0)
        ret = max(self.path)
        # print(self.path)
        return ret

    def dfs(self, begin: int, weight_tmp: int, value_tmp: int):
        if weight_tmp <= self.weight_max:
            self.path.append(value_tmp)
            # return self.maxvalue
        else:
            return
        for i in range(begin, len(self.weight_ls)):
            weight_tmp = weight_tmp + self.weight_ls[i]
            value_tmp = value_tmp + self.value_ls[i]
            self.dfs(begin, weight_tmp, value_tmp)
            weight_tmp = weight_tmp - self.weight_ls[i]
            value_tmp = value_tmp - self.value_ls[i]
            self.dfs(begin+1, weight_tmp, value_tmp)


if __name__ == '__main__':
    weight_max = int(input().strip())
    weight_ls = input().strip().split(",")
    weight_ls = [int(weight_ls[i]) for i in range(len(weight_ls))]
    value_ls = input().strip().split(",")
    value_ls = [int(value_ls[i]) for i in range(len(value_ls))]
    solution = solution(weight_max, weight_ls, value_ls)
    ret = solution.maxValue()
    print(ret)


