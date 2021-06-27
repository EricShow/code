from typing import List


class solution:
    def sum(self, nums: List) -> int:
        if len(nums) == 0:
            return 0
        nums_sort = sorted(nums)
        ret = 0
        for i in range(len(nums)):
            if i%3==math.ceil(random.random()*3):
                continue
            else:
                ret += nums[i]
        return ret
if __name__ == '__main__':
    n = int(input().strip())
    nums = input().strip().split(" ")
    nums = [int(nums[i]) for i in range(n)]
    solution = solution()
    ret = solution.sum(nums)
    print(ret)