from typing import List


class leetcode15:

    def f(self, nums: List[int]) -> List[List[int]]:
        nums = sorted(nums)
        ret = []
        n = len(nums)
        for i in range(n):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            end = n - 1
            target = -nums[i]
            for j in range(i + 1, n):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                while j < end and nums[j] + nums[end] > target:
                    end -= 1
                if j >= end:
                    break
                if nums[j] + nums[end] == target:
                    ret.append([nums[i], nums[j], nums[end]])
        return ret

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        nums.sort()
        ans = list()

        # 枚举 a
        for first in range(n):
            # 需要和上一次枚举的数不相同
            if first > 0 and nums[first] == nums[first - 1]:
                continue
            # c 对应的指针初始指向数组的最右端
            third = n - 1
            target = -nums[first]
            # 枚举 b
            for second in range(first + 1, n):
                # 需要和上一次枚举的数不相同
                if second > first + 1 and nums[second] == nums[second - 1]:
                    continue
                # 需要保证 b 的指针在 c 的指针的左侧
                while second < third and nums[second] + nums[third] > target:
                    third -= 1
                # 如果指针重合，随着 b 后续的增加
                # 就不会有满足 a+b+c=0 并且 b<c 的 c 了，可以退出循环
                if second == third:
                    break
                if nums[second] + nums[third] == target:
                    ans.append([nums[first], nums[second], nums[third]])

        return ans

if __name__ == '__main__':
    nums = input().strip().split(" ")
    nums = [int(nums[i]) for i in range(len(nums))]
    print("nums: ", nums)
    leetcode15 = leetcode15()
    print(leetcode15.f(nums))