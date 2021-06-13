from typing import List

#python的[]也是stack
class leetcode71:     # 括号的合理性
    def simplifyPath(self, path: str) -> str:
        r = []
        for s in path.split('/'):
            r = {'': r, '.': r, '..': r[:-1]}.get(s, r + [s])
        return '/' + '/'.join(r)
    def simplifyPath1(self, path: str) -> str:
        stack = []
        for p in path.split('/'):
            if p not in ['.', '..', '']:
                stack.append(p)
            elif p == '..' and stack:
                stack.pop()
        if len(stack) == 0:
            return "/"
        ret = ""
        for i in range(len(stack)):
            ret += "/" + stack[i]
        return ret

    def simplifyPath2(self, path: str) -> str:
        stack = []
        for p in path.split('/'):
            if p not in ['.', '..', '']:
                stack.append(p)
            elif p == '..' and stack:
                stack.pop()
        return f"/{'/'.join(stack)}"



if __name__ == '__main__':
    path = input()
    print("path: ", path)
    ls = path.strip("/").split("/")
    print("ls: ", ls)
    leetcode71 = leetcode71()
    ret = leetcode71.simplifyPath1(path)
    print("ret: ", ret)
