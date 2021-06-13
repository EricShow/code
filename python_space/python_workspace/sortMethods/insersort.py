from typing import List
import time


class charu:
    def insertsort(self, nums: List[int]) -> List[int]:
        start_time = time.time()
        n = len(nums)
        for i in range(1, n):
            if nums[i] < nums[i - 1]:
                # 暂存需要插入的数据
                temp = nums[i]
                j = i - 1
                # 一边比较一边后移
                while j >= 0 and nums[j] > temp:
                    nums[j + 1] = nums[j]
                    j -= 1
                # 跳出循环时表示j>0 或者nums[j] <= temp，那么应该插在后一个位置
                nums[j + 1] = temp
        end_time = time.time()
        print("运行时间: ", end_time - start_time)
        return nums

if __name__ == '__main__':
    nums = input().strip().split(" ")
    nums = [int(nums[i]) for i in range(len(nums))]
    charu = charu()
    ret = charu.insertsort(nums)
    print("ret: ", ret)