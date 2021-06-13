from typing import List


class leetcode344:

    def reverseString(self, s: List[str]) -> None:
        for i in range(len(s)//2):
            tmp = s[i]
            s[i] = s[len(s)-i-1]
            s[len(s)-i-1] = tmp

    def reverseString2(self, s: List[str]) -> None:
        s = s[::-1]