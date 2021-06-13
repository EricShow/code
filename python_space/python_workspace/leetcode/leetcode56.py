from typing import List


class leetcode56:

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            # 如果列表为空，或者当前区间与上一区间不重合，直接添加
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                # 否则的话，我们就可以与上一区间进行合并
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged
    def merge_sdl(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])

        merged = []
        for idx, interval in enumerate(intervals):
            # 如果列表为空，或者当前区间与上一区间不重合，直接添加
            if not merged or merged[-1][1] < intervals[idx][0]:
                merged.append(interval)
            else:
                # 否则的话，我们就可以与上一区间进行合并
                merged[-1][1] = max(merged[-1][1], intervals[idx][1])

        return merged
if __name__ == '__main__':

    intervals = [[1,3],[2,6],[8,10],[15,18]]
    print("intervals: ", intervals)
    #intervals = sorted(intervals)
    #intervals.sort()
    leetcode56 = leetcode56()
    ret = leetcode56.merge_sdl(intervals)
    print("ret: ", ret)