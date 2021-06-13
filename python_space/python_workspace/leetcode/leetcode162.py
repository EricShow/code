from typing import List


class leetcode162:
    def findPeakElement_sdl(self, nums: List[int]) -> int:
    #问题：nums数组一定会有峰值 -1根本不存在
        if len(nums)==1:
            return 0
        for i in range(len(nums)):
            if i==0 and nums[i]>nums[i+1]:
                return i
            elif i==len(nums)-2 and nums[i+1]>nums[i]:
                return i+1
            else:
                if nums[i]>nums[i-1] and nums[i]>nums[i+1]:
                    return i
        return -1
    def findPeakElement(self, nums: List[int]) -> int:
        for i in range(len(nums)-1):
            if nums[i]>nums[i+1]:
                return i
        return nums.__len__()-1

if __name__ == '__main__':
    nums = input().strip().split(" ")
    nums = [int(nums[i]) for i in range(len(nums))]
    print("nums: ", nums)
    leetcode162 = leetcode162()
    print(leetcode162.findPeakElement(nums))