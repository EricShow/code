class leetcode191:
    def hammingWeight(self, n:int)->int:
        return bin(n).count('1')
    def f(self, n:int)->int:
        return hex(n).count('a')
if __name__ == '__main__':
    n = int(input().strip())
    leetcode191 = leetcode191()
    print(hex(n))
    print(leetcode191.f(n))
