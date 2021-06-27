from typing import List


class arraySegmentation:

    def seg(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        tmp = nums[0]
        ret = []
        ls = [tmp]
        for i in range(1, n):
            if tmp == nums[i]:
                ls.append(tmp)
            else:
                ret.append(ls)
                tmp = nums[i]
                ls = [tmp]
        if ls != None:
            ret.append(ls)
        return ret

if __name__ == '__main__':
    arraySegmentation = arraySegmentation()
    # nums = input().strip().split(" ")
    # nums = [int(tmp) for tmp in nums]
    nums = [0,0,0,1,1,2,3,3,3,2,3,3,0,0]
    ret = arraySegmentation.seg(nums)
    print("ret: ", ret)