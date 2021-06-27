from typing import List


class leetcode139:

    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = [False]*(len(s)+1)
        dp[0] = True

        for i in range(len(s)):
            for j in range(i+1, len(s)+1):
                if dp[i] and (s[i:j] in wordDict):
                    dp[j] = True

        return dp[-1]

if __name__ == '__main__':
    #s = input().strip()
    #wordDict = input().strip().split(" ")
    s = "leetcode"
    wordDict = ['leet', 'code']
    print("wordDict: ", wordDict)
    leetcode139 = leetcode139()
    ret = leetcode139.wordBreak(s, wordDict)
    print("ret: ", ret)