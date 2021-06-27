from typing import List
import time

class xuanze:
    def selectsort(self, nums: List[int]) -> List[int]:
        start_time = time.time()
        n = len(nums)
        for i in range(n):
            minindex = i
            for j in range(i + 1, n):
                if nums[j] < nums[minindex]:
                    minindex = j
            if minindex != i:
                nums[i], nums[minindex] = nums[minindex], nums[i]
        end_time = time.time()
        print("运行时间: ", end_time - start_time)
        return nums

if __name__ == '__main__':
    nums = input().strip().split(" ")
    nums = [int(nums[i]) for i in range(len(nums))]
    xuanze = xuanze()
    ret = xuanze.selectsort(nums)
    print("ret: ", ret)
