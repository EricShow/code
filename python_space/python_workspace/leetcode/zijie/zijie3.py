import math

class solution:
    def countofseats(self, n: int):
        if n==1:
            return 1
        if n==2:
            return 4
        dp = [0]*(n+1)
        dp[0] = 0
        dp[1] = 1
        dp[2] = 4
        for i in range(3, n+1):
            dp[i] = dp[i//2]+dp[i-i//2]+3
        return dp[n]
if __name__ == '__main__':
    T = int(input().strip())
    solution = solution()
    for i in range(T):
        n = int(input())
        ret = solution.countofseats(n)
        print(ret)

#状态转移方程

#f[i]=f[i/2]+f[i-i/2]+3，然后f[1]=1,f[2]=4