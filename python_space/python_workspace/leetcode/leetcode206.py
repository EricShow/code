from ListNode import ListNode

#反转链表
class leetcode206:
    def reverseList(self, head: ListNode) -> ListNode:
        prev = None
        cur = head
        while cur != None:
            temp = cur.next
            cur.next = prev
            prev = cur
            cur = temp
        return prev


if __name__ == '__main__':
    input = input().strip().split(" ")
    input = [int(input[i]) for i in range(len(input))]
    print("input: ", input)
    ls = ListNode(0)
    cur = ls
    for i in range(len(input)):
        ls.next = ListNode(input[i])
        ls = ls.next
    ls = cur.next
    # while ls != None:
    #     print(ls.val, end="-->")
    #     ls = ls.next
    leetcode206 = leetcode206()
    prev = leetcode206.reverseList(ls)
    while prev != None:
        print(prev.val, end="-->")
        prev = prev.next
