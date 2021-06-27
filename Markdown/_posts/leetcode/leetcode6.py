class leetcode6:
    def convert_3(self, s: str, numRows: int) -> str:
        ls0 = ""
        ls1 = ""
        ls2 = ""
        ls3 = ""
        for i in range(len(s)):
            if i%4==0:
                ls0 += s[i]
            elif i%4==2:
                ls2 += s[i]
            else:
                ls1 += s[i]
        ret = ls0 + ls1 + ls2
        return ret

    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1: return s
        rows = [""] * numRows
        n = 2 * numRows - 2
        for i, char in enumerate(s):
            x = i % n
            rows[min(x, n - x)] += char
        ret = ""
        for i in range(len(rows)):
            ret += rows[i]
        #return "".join(rows)
        return ret
if __name__ == '__main__':
    s = input().strip()
    leetcode6 = leetcode6()
    print("ret: ", leetcode6.convert(s, 3))