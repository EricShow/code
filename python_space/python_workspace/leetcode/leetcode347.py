from typing import List
from collections import Counter


# 前K个高频元素
class leetcode347:

    def topKFrequent(self, nums: List[str], k:int) -> List[str]:
        l = {}
        for i in set(nums):
            l[i] = 0
        for i in nums:
            l[i] += 1
        l = dict(sorted(l.items(), key=lambda item: item[1], reverse=True))
        ll = []
        for key, value in l.items():
            ll.append(key)
            k -= 1
            if k == 0:
                break
        return ll
    def topKFrequent_count(self, nums: List[str], k:int) -> List[str]:
        ret = []
        for i in Counter(nums).most_common(k):
            ret.append(i[0])
        return ret
        #return [i[0] for i in Counter(nums).most_common(k)]
    def topKFrequent_str(self, nums: List[int], k:int) -> List[int]:
        l = {}
        for i in set(nums):
            l[i] = 0
        for i in nums:
            l[i] += 1
        l = dict(sorted(l.items(), key=lambda item: item[1], reverse=True))
        ll = []
        for key, value in l.items():
            ll.append(key)
            k -= 1
            if k == 0:
                break
        return ll
    def topKFrequent_count_str(self, nums: List[int], k:int) -> List[int]:
        ret = []
        for i in Counter(nums).most_common(k):
            ret.append(i[0])
        return ret
        #return [i[0] for i in Counter(nums).most_common(k)]
if __name__ == '__main__':
    k = int(input().strip())
    nums = input().strip().split(" ")
    nums = [num for num in nums]
    # nums = [int(num) for num in nums]
    print("nums: ", nums)
    print("top k: ", k)
    leetcode347 = leetcode347()
    ret = leetcode347.topKFrequent_count(nums, k)
    print("ret: ", ret)
