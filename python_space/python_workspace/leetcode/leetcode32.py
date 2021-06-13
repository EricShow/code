from typing import List


class leetcode32:

    def longestValidParentheses_sdl(self, s: str) -> int:
    # 通过率91/231
        if not s:
            return 0
        count = 0
        while '()' in s or '[]' in s or '{}' in s:
            for index in ['()', '[]', '{}']:
                if index in s:
                    count1 = 0
                    idx = s.find(index)
                    while idx!=-1:
                        idx= s.find(index, idx+2, idx+4)
                        count1 += 1
                    count += 2*count1
                    s = s.replace(index, '')
        return count
    def longestValidParentheses(self, s: str) -> int:
        maxans = 0
        dp = [0]*len(s)
        for i in range(1,len(s)):
            if s[i]==')':
                if s[i-1]=='(':
                    dp[i] = dp[i-2]+2 if i>=2 else 2
                    # 二者等价，python的三目运算是通过if else实现的
                    # if i>=2:
                    #     dp[i] = dp[i-2]+2
                    # else:
                    #     dp[i] = 2
                elif i-dp[i-1] > 0 and s[i-dp[i-1]-1]=='(':
                    dp[i] = dp[i-1] + dp[i-dp[i-1]-2] + 2 if i-dp[i-1]>=2 else dp[i-1] + 2
                    # if i-dp[i-1]>=2:
                    #     dp[i] = dp[i-1] + dp[i-dp[i-1]-2] + 2
                    # else:
                    #     dp[i] = dp[i-1] + 2
                maxans = max(maxans, dp[i])
        return maxans
if __name__ == '__main__':
    s = input().strip()
    print("s: ", s)
    leetcode32 = leetcode32()
    ret = leetcode32.longestValidParentheses(s)
    print("ret: ", ret)