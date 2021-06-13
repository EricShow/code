from typing import List


class leetcode67:

    def addBinary(self, a: str, b: str) -> str:
        len_a = len(a)
        len_b = len(b)
        i_a = len_a-1
        i_b = len_b-1
        carry = 0
        ret = ""
        while i_a>=0 or i_b>=0:
            if i_a >= 0:
                a_val = int(a[i_a])
                i_a -= 1
            else:
                a_val = 0
            if i_b >= 0:
                b_val = int(b[i_b])
                i_b -= 1
            else:
                b_val = 0
            sum = a_val + b_val + carry
            yushu = sum % 2
            carry = sum // 2
            ret = ret + str(yushu)
        if carry == 1:
            ret = ret + str(carry)
        return ret[::-1]

if __name__ == '__main__':
    a = "11"
    b = "1"
    # print字符串输出，print("%d"% 20,"%o" % 10)
    leetcode67 = leetcode67()
    ret = leetcode67.addBinary(a, b)
    print("ret: ", ret)