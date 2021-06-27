from typing import List


class solution:

    def shortestpath(self, n: int, visited_count:int, visited_ls: List[int], edge: List[List[int]]) -> int:
        # for i in range(n):
        #     for j in range(n):
        pathes = self.allpath(edge)
        if n-visited_count == 1:
            for i in range(n):
                if i not in visited_ls:
                    for path in pathes:
                        if i in pathes:
                            return n
        #
        #
        #
        # for visit in visited_ls:
        #     visited_path = [visit]
        #     for path in pathes:
        #         if visit in path:
        #             if path[0]==visit:
        #                 visited_path.append(path[1])
        #             else:
        #                 visited_path.append(path[0])
        # #
        # for i in range(len(path)-1):
        #     tmp = []
        #     if path[i][-1] == path[i+1][0]:
        #         #temp = path[i].append(path[i+1][1])
        #         tmp = [path[i][0],path[i][1], path[i+1][1]]
        #         path.append(tmp)
        # print("path: ", path)
        return visited_count+2
    def allpath(self, edge: List[List[int]]) -> List:
        path = []
        for i in range(n):
            for j in range(i+1, n):
                if edge[i][j]==1:
                    path.append([i,j])
        return path


if __name__ == '__main__':
    # n, visited_count = 4,3
    # edge = [[0,1,0,0],[1,0,1,1],[0,1,0,0],[0,1,0,0]]
    # visited_ls = [0,2,3]



    n, visited_count= input().strip().split(" ")
    n = int(n)
    visited_count = int(visited_count)
    #print("n: ", n)
    #print("visited_count: ", visited_count)
    edge = []
    for i in range(n):
        tmp = input().strip().split(" ")
        temp = [int(a) for a in tmp]
        edge.append(temp)
    #print("edge: ", edge)
    ls = input().strip().split(" ")
    visited_ls = [int(ls[i]) for i in range(len(ls))]
    #print("visited_ls: ", visited_ls)
    solution = solution()

    ret = solution.shortestpath(n, visited_count, visited_ls, edge)
    print(ret)


#参考答案
def run(inf, bad, stack, n=0):
    if n == len(bad):
        return len(stack)

    res = len(inf)
    for i, f in enumerate(inf[stack[-1]]):
        if i not in stack and f == 1:
            stack.append(i)
            if i in bad:
                n += 1
            res = min(res, run(inf, bad, stack, n))
            print(stack, res)
            stack.pop()
            if i in bad:
                n -= 1
    return res

u1 = bad[0]
stack = [u1]
n = 1
run(inf, bad, stack, n)