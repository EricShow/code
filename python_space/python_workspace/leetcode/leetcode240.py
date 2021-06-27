from typing import List


class leetcode240:

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        row_l = 0
        row_r = len(matrix)
        col_l = 0
        col_r = len(matrix[0])
        return self.dfs(matrix, row_l, row_r, col_l, col_r, target)
    def dfs(self, matrix: List[List[int]], row_start:int , row_end:int, col_start:int , col_end:int, target: int) -> bool:
        row_l = row_start
        row_r = row_end
        col_l = col_start
        col_r = col_end
        row_mid = row_l + (row_r - row_l) // 2
        col_mid = col_l + (col_r - col_l) // 2
        if matrix[row_mid][col_mid] == target:
            return True
        elif matrix[row_mid][col_mid] > target:
            return self.dfs(matrix, row_l, row_mid, col_l, col_mid, target)
        elif matrix[row_mid][col_mid] < target:
            return self.dfs(matrix, row_l, row_mid, col_mid, col_end, target) or \
            self.dfs(matrix, row_mid, row_end, col_l, col_mid, target) or \
            self.dfs(matrix, row_mid, row_end, col_mid, col_end, target)
        else:
            return False

    def searchMatrix_official(self, matrix: List[List[int]], target: int) -> bool:
        def DFS(nx, ny):
            if nx < 0 or ny >= len(matrix[0]):
                return False
            if matrix[nx][ny] == target:
                return True
            if matrix[nx][ny] > target:
                return DFS(nx - 1, ny)
            else:
                return DFS(nx, ny + 1)

        return DFS(len(matrix) - 1, 0)

if __name__ == '__main__':
    #matrix = [[1, 4, 7, 11, 15], [2, 5, 8, 12, 19], [3, 6, 9, 16, 22], [10, 13, 14, 17, 24], [18, 21, 23, 26, 30]]
    matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]
    target = 20
    leetcode240 = leetcode240()
    print("ret: ", leetcode240.searchMatrix(matrix, target))