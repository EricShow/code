from typing import List
import time


class dui:

    def heapsort(self, nums):
        n = len(nums)
        # 建立大顶堆
        for i in range(n // 2 - 1, -1, -1):
            self.heapAjust(nums, i, n - 1)
        for i in range(n - 1, -1, -1):
            # 堆顶元素为最大值，交换至数组末尾
            nums[0], nums[i] = nums[i], nums[0]
            # 调整剩下数组仍为大顶堆
            self.heapAjust(nums, 0, i - 1)
        return nums

    def heapAjust(self, nums, start, end):
        temp = nums[start]
        # 记录较大的那个孩子下标
        child = 2 * start + 1
        while child <= end:
            # 比较左右孩子，记录较大的那个
            if child + 1 <= end and nums[child] < nums[child + 1]:
                # 如果右孩子比较大，下标往右移
                child += 1
            # 如果根已经比左右孩子都大了，直接退出
            if temp >= nums[child]:
                break
            # 如果根小于某个孩子,将较大值提到根位置
            nums[start] = nums[child]
            # nums[start], nums[child] = nums[child], nums[start]
            # 接着比较被降下去是否符合要求，此时的根下标为原来被换上去的那个孩子下标
            start = child
            # 孩子下标也要下降一层
            child = child * 2 + 1
        # 最后将一开始的根值放入合适的位置(如果前面是交换，这句就不要)
        nums[start] = temp

if __name__ == '__main__':
    nums = input().strip().split(" ")
    nums = [int(nums[i]) for i in range(len(nums))]
    print("nums: ", nums)
    dui = dui()
    ret = dui.heapsort(nums)
    print("ret: ", ret)