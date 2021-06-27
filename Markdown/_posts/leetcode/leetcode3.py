from locale import str

class leetcode3:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s)<1:
            return len(s)
        hashtable = {}
        left, max_len = 0, 0
        for right in range(len(s)):
            cur_char = s[right]
            if cur_char in hashtable:
                if hashtable[cur_char] +1 >= left:
                    left = hashtable[cur_char] + 1
            #hashtable增加元素的方式 hashtable[key] = val
            hashtable[cur_char] = right
            max_len = max(max_len, right-left+1)
        return max_len
    def sdl_function(self, s: str) -> int:
        len_s = len(s)
        ans = 0
        hashmap = dict()
        start, end = 0, 0
        for i in range(len_s):
            tmp = s[i]
            if hashmap.__contains__(tmp):
                start = max(hashmap.get(tmp), start)
            ans = max(ans, end - start + 1)
            hashmap[tmp] = end + 1
            end += 1
        return ans
if __name__ == '__main__':
    str = input()
    leetcode3 = leetcode3()
    ret = leetcode3.lengthOfLongestSubstring(str)
    ret = leetcode3.sdl_function(str)
    print("")
    print(ret)