from typing import List

# 经典回溯问题  组合总和II
# 输入: candidates = [10,1,2,7,6,1,5], target = 8
# 输出: ret:  [[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]]
class leetcode40_self:

    def __init__(self, canditates, target):
        self.candidates = canditates
        self.target = target
        self.size = len(canditates)
        self.res = []
    def combinationSum2(self) -> List[List[int]]:
        if self.size == 0:
            return []
        self.candidates.sort()
        self.dfs(0, [], self.target)
        return self.res
    def dfs(self, begin: int, path: List, residue: int):
        if residue == 0:
            self.res.append(path[:])
            return
        for index in range(begin, self.size):
            if candidates[index] > residue:
                break
            if index > begin and candidates[index - 1] == candidates[index]:
                continue
            path.append(candidates[index])
            self.dfs(index + 1, path, residue - self.candidates[index])
            path.pop()


class leetcode40:

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        def dfs(begin, path, residue):
            if residue == 0:
                res.append(path[:])
                return

            for index in range(begin, size):
                if candidates[index] > residue:
                    break

                if index > begin and candidates[index - 1] == candidates[index]:
                    #去重复
                    continue

                path.append(candidates[index])
                dfs(index + 1, path, residue - candidates[index])
                path.pop()

        size = len(candidates)
        if size == 0:
            return []

        candidates.sort()
        res = []
        dfs(0, [], target)
        return res

class leetcode40_sdl:
#组合总和40  回溯算法
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        size = len(candidates)
        if size == 0:
            return []

        candidates.sort()
        res = []
        self.dfs(candidates, 0, [], target, res)
        return res

    def dfs(self, candidates: List, begin: int, path: List, residue: int, res: List):
        size = len(candidates)
        if residue == 0:
            res.append(path[:])
            return

        for index in range(begin, size):
            if candidates[index] > residue:
                break

            if index > begin and candidates[index - 1] == candidates[index]:
                # 去重复
                continue

            path.append(candidates[index])
            self.dfs(candidates, index + 1, path, residue - candidates[index], res)
            path.pop()

if __name__ == '__main__':

    print("candidates: ")
    candidates = input().strip().split(" ")
    candidates = [int(candidates[i]) for i in range(len(candidates))]
    print("target: ")
    target = int(input().strip())

# leetcode40_self
    print("leetcode40_self")
    leetcode40_self = leetcode40_self(candidates, target)
    ret = leetcode40_self.combinationSum2()
    print("ret: ", ret)
# leetcode40
    print("leetcode40")
    leetcode40 = leetcode40()
    ret= leetcode40.combinationSum2(candidates, target)
    print("ret: ", ret)
# leetcode40_sdl
    print("leetcode40_sdl")
    leetcode40_sdl = leetcode40_sdl()
    ret= leetcode40_sdl.combinationSum2(candidates, target)
    print("ret: ", ret)