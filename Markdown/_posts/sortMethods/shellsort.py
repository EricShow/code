from typing import List
import time
import math


class hill:
    def shellsort(self, array: List[int]) -> List[int]:
        interval = int(len(array) / 3)
        while interval > 0:
            for i in range(interval, len(array)):
                cur_index = i - interval
                while cur_index >= 0 and array[cur_index] > array[cur_index + interval]:
                    array[cur_index + interval], array[cur_index] = array[cur_index], array[cur_index + interval]
                    cur_index -= interval
            interval = int(interval / 3)
        return array

if __name__ == '__main__':
    nums = input().strip().split(" ")
    nums = [int(nums[i]) for i in range(len(nums))]
    print("nums: ", nums)
    hill = hill()
    ret = hill.shellsort(nums)
    print("ret: ", ret)
