from typing import List


class merge_sdl:


    def merge(self, ls1:List[int], ls2: List[int]) -> List[int]:
        n1 = len(ls1)
        n2 = len(ls2)
        ret = [0] * (n1+n2)
        l1 = 0
        l2 = 0
        count = 0
        while l1<n1 and l2<n2:
            tmp1 = ls1[l1]
            tmp2 = ls2[l2]
            if tmp1<=tmp2:
                ret[count] = tmp1
                l1 += 1
            else:
                ret[count] = tmp2
                l2 += 1
            count += 1
        if l1 >= n1:
            for i in range(l2, n2):
                ret[count] = ls2[i]
                count += 1
        else:
            for i in range(l1, n1):
                ret[count] = ls1[i]
                count += 1
        return ret

if __name__ == '__main__':
    ls1 = [3,7,8,9,12]
    ls2 = [5,6,10,13,25,30]
    merge_sdl = merge_sdl()
    ret = merge_sdl.merge(ls1, ls2)
    print(ret)