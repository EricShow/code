class leetcode5_2:
# input:babad
# output: bab
    def longestPalindrome(self, s:str) ->str:
        ret = s[0:1]
        for i in range(len(s)):
            ret1 = self.f(s,i,i)
            if len(ret1) > len(ret):
                ret = ret1
            ret2 = self.f(s,i,i+1)
            if len(ret2) > len(ret):
                ret = ret2
        return ret
    def f(self, s:str, start:int, end:int) ->str:
        n = len(s)
        left = start
        right = end
        ret = s[left:left+1:1]
        while left>=0 and right<n and left<=right and s[left]==s[right]:
            ret = s[left:right+1]
            left -= 1
            right +=1
        return ret

if __name__ == '__main__':

    s = input().strip()
    print("s: ", s)
    leetcode5_2 = leetcode5_2()
    ret = leetcode5_2.longestPalindrome(s)
    print(ret)