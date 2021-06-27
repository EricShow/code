class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class solution:

    def inverse(self, root:TreeNode) ->TreeNode:
        if root==None:
            return root
        left = root.right
        right = root.left
        root.left = left
        root.right = right
        self.inverse(root.left)
        self.inverse(root.right)
        return root

if __name__ == '__main__':
    ts = TreeNode(0)
    ts.left = TreeNode(1)
    ts.right = TreeNode(2)
    ts.left.left = TreeNode(3)
    ts.left.right = TreeNode(4)
    ts.right.left = TreeNode(5)

    solution = solution()
    ret = solution.inverse(ts)
    print(ret)

