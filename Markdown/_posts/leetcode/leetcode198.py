from typing import List
import numpy as np
# 只需要考虑每天能偷到的最大值，不用考虑偷不偷
class leetcode198:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        size = len(nums)
        if size == 1:
            return nums[0]
        dp = [0] * size
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in range(2, size):
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1])

        return dp[size - 1]
if __name__ == '__main__':
    nums = input().strip().split(" ")
    nums = [int(nums[i]) for i in range(len(nums))]
    leetcode198 = leetcode198()
    ret = leetcode198.rob(nums)
    print(ret)
