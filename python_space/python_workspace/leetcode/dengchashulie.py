from typing import List


class dengchashulie:


    def dengcha(self, nums: List[int]) -> bool:
        n = len(nums)
        if n==0 or n==1:
            return True
        d = nums[1]-nums[0]
        for i in range(1, n-1):
            tmp = nums[i+1] - nums[i]
            if tmp != d:
                return False
            
        return True

if __name__ == '__main__':
    #nums = [5, 3, 2, 1, 7, 6, 4, 8, 10, 9]
    nums = [1,2,3,4,5,6,7,8]
    dengchashulie = dengchashulie()
    print(dengchashulie.dengcha(nums))
