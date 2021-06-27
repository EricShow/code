from typing import List


class leetcode44:
    def findNthDigit(self, n: int) -> int:
        digit, start, count = 1, 1, 9
        while n > count: # 1.
            n -= count
            start *= 10
            digit += 1
            count = 9 * start * digit
        num = start + (n - 1) // digit # 2.
        return int(str(num)[(n - 1) % digit])

if __name__ == '__main__':
    n = int(input().strip())
    leetcode44 = leetcode44()
    ret = leetcode44.findNthDigit(n)
    print("ret: ", ret)