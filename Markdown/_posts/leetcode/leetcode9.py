class leetcode9:
    def isPalindrome(self, x: int) -> bool:
        if x<0:
            return False
        cur = 0
        num = x
        while num != 0:
            cur = cur*10 + num%10
            num = num // 10
        return cur == x
if __name__ == '__main__':
    x = int(input().strip())
    leetcode9 = leetcode9()
    print("return: ",leetcode9.isPalindrome(x))
