from typing import List
# 插入区间

class leetcode57:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        left, right = newInterval
        placed = False
        ans = []
        for li, ri in intervals:
            if li>right:
                # 在插入区间的右侧且无交集
                if not placed:
                    ans.append([left, right])
                    placed = True
                # 这种情况说明无交集，且这个new还没加入，所以可以append new 以及 append([li, ri])
                # 后面的一定都满足li>right, 因为已经place了，所以直接插入
                ans.append([li, ri])
            elif ri<left:
                # 在插入区间的左侧且无交集
                ans.append([li, ri])
            else:
                # 与插入区间有交集，计算它们的并集
                left = min(left, li)
                right = max(right, ri)
        if not placed:
            ans.append([left, right])
        return ans

if __name__ == '__main__':
    # intervals = [[1, 3], [6, 9]]
    # newInterval = [2, 5]
    count = int(input().strip())
    intervals = []
    for i in range(count):
        interval = input().strip().split(" ")
        interval = [int(tmp) for tmp in interval]
        intervals.append(interval)
    print("intervals: ", intervals)
    newInterval = input().strip().split(" ")
    newInterval = [int(newInterval[i]) for i in range(len(newInterval))]
    print("newInterval: ", newInterval)
    leetcode57 = leetcode57()
    ret = leetcode57.insert(intervals, newInterval)
    print("ret: ", ret)