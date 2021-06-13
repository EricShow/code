from ListNode import ListNode
from TreeNode import TreeNode
class leetcode109:
    def sortedListToBST(self, head: ListNode) -> TreeNode:
        return self.buildTree(head, None)
    def findMid(self, left: ListNode, right: ListNode)-> ListNode:
        fast = slow = left
        while fast!=right and fast.next!=right:
        # 注意二者的顺序，一定是先判断fast再判断fast.next
            fast = fast.next.next
            slow = slow.next
        return slow
    def buildTree(self, left:ListNode, right:ListNode):
        if left==right:
            return None
        mid = self.findMid(left, right)
        root = TreeNode(mid.val)
        root.left = self.buildTree(left, mid)
        root.right = self.buildTree(mid.next, right)
        return root
if __name__ == '__main__':
    list = input()
    list = list.strip().split(" ")
    list = [int(list[i]) for i in range(len(list))]
    print("list: ", list)
    ls = ListNode(0)
    cur = ls
    for i in range(len(list)):
        ls.next = ListNode(list[i])
        ls = ls.next
    ls = cur.next
    leetcode109 = leetcode109()
    ret = leetcode109.sortedListToBST(ls)
    print("ret: ", ret)