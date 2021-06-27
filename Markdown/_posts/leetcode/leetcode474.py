from typing import List

# 一和零
# 没有完全理解的代码
class leetcode474:
    def findMaxForm(self, strs: List[str], m: int, n: int) -> int:
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        #初始化一个数组6*5 dp:  [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        for s in strs:
            a, b = self.count01(s)
            for i in range(m, a - 1, -1):
                for j in range(n, b - 1, -1):
                    dp[i][j] = max(dp[i][j], dp[i - a][j - b] + 1)
        return dp[m][n]

    def count01(self, s: str):
        ret = [0] * 2
        for i in range(len(s)):
            if s[i] == '0':
                ret[0]+=1
            else:
                ret[1]+=1
        return ret

if __name__ == '__main__':
    strs = input().strip().split(" ")
    print("strs: ", strs)
    print("m = ")
    m = int(input().strip())
    print("n = ")
    n = int(input().strip())
    leetcode474 = leetcode474()
    ret = leetcode474.findMaxForm(strs, m, n)
    print("ret: ", ret)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    print("dp: ", dp)

