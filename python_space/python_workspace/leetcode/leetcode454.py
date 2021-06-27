from typing import List


class leetcode454:


    def fourSumCount(self, nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
        ret = []
        for i in nums1:
            for j in nums2:
                for k in nums3:
                    for m in nums4:
                        tmp = i+j+k+m
                        if tmp == 0:
                            ret.append([i, j, k, m])
        return len(ret)

if __name__ == '__main__':
    nums1 = [1, 2]
    nums2 = [-2, -1]
    nums3 = [-1, 2]
    nums4 = [0, 2]

    leetcode454 = leetcode454()
    ret = leetcode454.fourSumCount(nums1, nums2, nums3, nums4)
    print("ret: ", ret)