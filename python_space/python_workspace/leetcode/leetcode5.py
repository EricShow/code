
class leetcode5:
    # 通过率99/176
    def longestPalindrome_sdl_err(self, s: str) -> str:
        ret = s[0:1]
        for i in range(len(s)):
            if i==0 or i==len(s)-1:
                continue
            else:
                left = i
                right = i
                ans = s[left]
                while(left>=0 and s[left]==s[right] and right<len(s)-1):
                    ans = s[left:right+1]
                    left = left-1
                    right = right+1
                if len(ans)>len(ret):
                    ret = ans
        for i in range(len(s)-1):
            if(s[i]==s[i+1]):
                if i==0:
                    ans = s[0:1]
                elif i==len(s)-2:
                    ans = s[i:i+1]
                else:
                    left = i
                    right = i+1
                    ans = s[left:right+1]
                    while (left >= 0 and s[left] == s[right] and right < len(s) - 1):
                        ans = s[left:right + 1]
                        left = left - 1
                        right = right + 1
                if len(ans)>len(ret):
                    ret = ans
            else:
                continue
        return ret

#最长回文子串
    def longestPalindrome_sdl(self, s: str) -> str:
        start, end =0,0
        for i in range(len(s)):
            left1, right1 = self.f(s, i, i)
            left2, right2 = self.f(s, i, i+1)
            if(right1-left1>end-start):
                end = right1
                start = left1
            if(right2-left2>end-start):
                end = right2
                start = left2
        return s[start:end+1]
    def f(self, s, left ,right):
        while(left>=0 and right<len(s) and s[left]==s[right]):
            left -= 1
            right += 1
        return left+1, right-1





# 官方题解处理中心是一个还是两个的情况：
#   中心是一个或者两个都算一下
    def expandAroundCenter(self, s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return left + 1, right - 1

    def longestPalindrome(self, s: str) -> str:
        start, end = 0, 0
        for i in range(len(s)):
            left1, right1 = self.expandAroundCenter(s, i, i)
            left2, right2 = self.expandAroundCenter(s, i, i + 1)
            if right1 - left1 > end - start:
                start, end = left1, right1
            if right2 - left2 > end - start:
                start, end = left2, right2
        return s[start: end + 1]

if __name__ == '__main__':
    s = input().strip()
    leetcode5 = leetcode5()
    ret = leetcode5.f2(s)
    print("")
    print("return: ", ret)