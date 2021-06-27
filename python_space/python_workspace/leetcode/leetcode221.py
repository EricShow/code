from typing import List

class leetcode221:

    def maxSquare(self, matrix: List[List[str]]) -> int:
        rows = len(matrix)
        cols = len(matrix[0])
        ls = []
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j]=='0':
                    continue
                max_row = 0
                max_col = 0
                for k in range(i, rows):
                    if matrix[k][j]=='1':
                        max_row += 1
                    else:
                        break
                for k in range(j, cols):
                    if matrix[i][k]=='1':
                        max_col += 1
                    else:
                        break
                ls.append([i,j,max_row,max_col])
        ret = 1
        for ele in ls:
            for i in range(ele[0],ele[0]+ele[2]):
                for j in range(ele[1], ele[0]+ele[2]):
                    if matrix[i][j] != '1':
                        break
                    else:
                        row = max(i - ele[0]+1, 1)
                        col = max(j - ele[1]+1, 1)
            ret = max(min(row,col), ret)
        return ret*ret

    def f(self, matrix: List[List[str]]) -> int:
        if len(matrix)==0 or len(matrix[0])==0:
            return 0

        maxSide = 0
        rows, cols = len(matrix), len(matrix[0])
        dp = [[0]*cols for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                if matrix[i][j]=='1':
                    if i==0 or j==0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    maxSide = max(maxSide, dp[i][j])
        return maxSide*maxSide

    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return 0

        maxSide = 0
        rows, columns = len(matrix), len(matrix[0])
        dp = [[0] * columns for _ in range(rows)]
        for i in range(rows):
            for j in range(columns):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    maxSide = max(maxSide, dp[i][j])

        maxSquare = maxSide * maxSide
        return maxSquare


if __name__ == '__main__':
    matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
    leetcode221 = leetcode221()
    ret = leetcode221.f(matrix)
    print("ret: ", ret)