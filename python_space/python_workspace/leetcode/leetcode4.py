from typing import List


class leetcode4:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        num = nums1 + nums2
        num = sorted(num)
        if (len(num) % 2 == 1):
            return float(num[len(num) // 2])
        else:
            return float(num[len(num) // 2] + num[len(num) // 2 - 1]) / 2


if __name__ == '__main__':
    print("nums1 = :")
    nums1 = input()
    nums1 = nums1.strip().split(" ")
    nums1 = [int(nums1[i]) for i in range(len(nums1))]

    print("nums2 = :")
    nums2 = input()
    nums2 = nums2.strip().split(" ")
    nums2 = [int(nums2[i]) for i in range(len(nums2))]

    leetcode4 = leetcode4()
    ret = leetcode4.findMedianSortedArrays(nums1, nums2)
    print("ret: ", ret)
