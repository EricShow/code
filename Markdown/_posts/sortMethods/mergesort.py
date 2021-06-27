from typing import List
import time
import math

import numpy as np


class guibing:
    # 归并排序 思想：先递归分解数组，再合并数组
    def mergesort(self, nums: List[int]) -> List[int]:
        n = len(nums)
        if n <= 1:
            return nums
        mid = n // 2
        left_li = self.mergesort(nums[:mid])
        right_li = self.mergesort(nums[mid:])
        left_pointer, right_pointer = 0, 0
        result = []
        while left_pointer < len(left_li) and right_pointer < len(right_li):
            if left_li[left_pointer] <= right_li[right_pointer]:
                result.append(left_li[left_pointer])
                left_pointer += 1
            else:
                result.append(right_li[right_pointer])
                right_pointer += 1
        result += left_li[left_pointer:]
        result += right_li[right_pointer:]
        return result


if __name__ == '__main__':
    nums = input().strip().split(" ")
    nums = [int(nums[i]) for i in range(len(nums))]
    helparr = np.zeros_like(nums).tolist()
    # print("helparr: ", helparr)
    guibing = guibing()
    ret = guibing.mergesort(nums)
    print("ret: ", ret)