from typing import List

#岛屿数量
class leetcode200:

    def dfs_sdl(self, grid: List[List[str]], i: int, j: int):
        nr, nc = len(grid), len(grid[0])
        grid[i][j] = "0"
        for x,y in [(i-1,j), (i+1,j), (i, j-1), (i, j+1)]:
            if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
                self.dfs_sdl(grid, x, y)
    def numIslands_sdl(self, grid: List[List[str]]) -> int:
        nr, nc = len(grid), len(grid[0])
        if nr == 0:
            return 0
        numIslands = 0
        for i in range(nr):
            for j in range(nc):
                if grid[i][j]=="1":
                    numIslands += 1
                    self.dfs_sdl(grid, i, j)
        return numIslands

    def dfs(self, grid, r, c):
        grid[r][c] = 0
        nr, nc = len(grid), len(grid[0])
        for x, y in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if 0 <= x < nr and 0 <= y < nc and grid[x][y] == "1":
                self.dfs(grid, x, y)

    def numIslands(self, grid: List[List[str]]) -> int:
        nr = len(grid)
        if nr == 0:
            return 0
        nc = len(grid[0])

        num_islands = 0
        for r in range(nr):
            for c in range(nc):
                if grid[r][c] == "1":
                #等于1的就表示岛屿+1, 同时我们需要把和这个岛屿相连的置0
                    num_islands += 1
                    self.dfs(grid, r, c)

        return num_islands



if __name__ == '__main__':
    grid =  [["1","1","1","1","0"],
             ["1","1","0","1","0"],
             ["1","1","0","0","0"],
             ["0","0","0","1","0"]]
    # 键盘输入grid
    # grid = []
    # row = int(input().strip())
    # for i in range(row):
    #     col = input().strip().split(" ")
    #     grid.append(col)
    print("grid", grid)
    leetcode200 = leetcode200()
    ret = leetcode200.numIslands_sdl(grid)
    print("ret: ", ret)