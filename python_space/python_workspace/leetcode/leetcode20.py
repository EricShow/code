from typing import List


class leetcode20:

# 有效的括号
# 为什么不能在while内部加if
# my_str = 'I\'m a student'
# my_str = "I'm a student"

    def isValid(self, s: str) -> bool:
        while ('()' in s) or ('[]' in s) or ('{}' in s):
            s = s.replace('()', '')
            s = s.replace("{}", '')
            s = s.replace('[]', '')
        return s == ""

if __name__ == '__main__':
    s = input().strip()
    print("s: ", s)
    leetcode20 = leetcode20()
    ret = leetcode20.isValid(s)
    print("ret: ", ret)