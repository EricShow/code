from typing import List


class leetcode22:
    def generateParenthesis(self, n: int) -> List[str]:
        ls = "("
        ret = []
        self.f(n-1,n,ls,ret)
        return ret
    def f(self, left: int, right: int, ls: str, ret: List[str]):
        if left==0 and right==0:
            ret.append(ls)
            return
        if right < left:
            return
        if left > 0:
            self.f(left-1, right, ls+'(', ret)
        #问题：为什么这样写能保证不会只出现((()))，因为在left!=0时，执行到((()))时，return了，return后就能顺利执行到left!=0增加right的过程
        if right > 0:
            self.f(left, right-1, ls+')', ret)

if __name__ == '__main__':
    n = int(input().strip())
    leetcode22 = leetcode22()
    ret = leetcode22.generateParenthesis(n)
    print("len(ret):", len(ret))
    print(ret)