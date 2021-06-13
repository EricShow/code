# 排序算法总结
from typing import List
import time

class quicksort:
    def quicksort(self, nums: List[int], low: int, high: int) -> List[int]:
        start_time = time.time()
        if low < high:
            pivot = self.patition(nums, low, high)
            self.quicksort(nums, low, pivot - 1)
            self.quicksort(nums, pivot + 1, high)
        end_time = time.time()
        print("运行时间: ", end_time - start_time)
        return nums
        # 非递归实现

    def quicksort1(self, nums, low, high):
        para_stack = []
        if low < high:
            para_stack.append(low)
            para_stack.append(high)
            while len(para_stack) > 0:
                high = para_stack.pop()
                low = para_stack.pop()
                mid = self.patition(nums, low, high)
                if mid - 1 > low:
                    para_stack.append(low)
                    para_stack.append(mid - 1)
                if mid + 1 < high:
                    para_stack.append(mid + 1)
                    para_stack.append(high)

        return nums

    def patition(self, nums, low, high):
        # 以第一个元素作为枢轴值
        base = nums[low]
        while low < high:
            # high指针从后往前移，找到比枢轴值小的数字
            while low < high and nums[high] >= base:
                high -= 1
            # 将找到的数字复制至低位
            nums[low] = nums[high]
            # low指针从前往后移，找到比枢轴值大的数字
            while low < high and nums[low] <= base:
                low += 1
            # 将找到的数字复制至高位
            nums[low] = nums[low]
        # 跳出循环时，两指针重合，将枢轴值放入
        nums[low] = base
        return low


if __name__ == '__main__':
    nums = input().strip().split(" ")
    nums = [int(nums[i]) for i in range(len(nums))]
    print("nums: ", nums)
    quicksort = quicksort()
    ret = quicksort.quicksort(nums, 0, len(nums) - 1)
    print("ret: ", ret)
