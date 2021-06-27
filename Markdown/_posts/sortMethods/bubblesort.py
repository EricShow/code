from typing import List
import time

class maopao:
    def bubblesort(self, nums: List[int]) -> List[int]:
        start_time = time.time()
        n = len(nums)
        changed = True
        for i in range(n):
            # 增加标志位，如果上一次排序没有交换，说明已经排序成功，提前结束
            if changed:
                changed = False
                # 每此排序之后最大值会沉到最后，不参与下一轮比较
                for j in range(0, n - 1 - i):
                    if nums[j] > nums[j + 1]:
                        changed = True
                        # 将较大值甩到后面
                        nums[j], nums[j + 1] = nums[j + 1], nums[j]
        end_time = time.time()
        print("运行时间: ", end_time - start_time)
        return nums

if __name__ == '__main__':
    nums = input().strip().split(" ")
    nums = [int(nums[i]) for i in range(len(nums))]
    maopao = maopao()
    ret = maopao.bubblesort(nums)
    print("ret: ", ret)
