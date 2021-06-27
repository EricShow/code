from typing import List
class leetcode1:
    def twoSum(nums: List[int], target: int) -> List[int]:
        hashmap = dict()
        for i, num in enumerate(nums):
            if hashmap.get(target - num) is not None:
                return [hashmap.get(target - num), i]
            hashmap[num] = i
if __name__ == '__main__':
    print("输入列表")
    x = input()
    xlist = x.split(",")
    xlist = [int(xlist[i]) for i in range(len(xlist))]
    print("输入target")
    target = int(input())
    a = leetcode1.twoSum(xlist, target)
    print("a: ", a)
