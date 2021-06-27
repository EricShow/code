from typing import List


class solution:

    def search7(self, s: List[str]) ->int:
        count = 0
        for i in range(len(s)):
            if self.isSeven(s[i]):
                count += 1
        return count
    def isSeven(self, s: str) -> bool:
        num = int(s)
        if num%7 == 0:
            return True
        for i in range(len(s)):
            if i == '7':
                return True
        return False



if __name__ == '__main__':
    num = input().strip().split(" ")
    solution = solution()
    ret = solution.search7(num)
    print("ret: ", ret)