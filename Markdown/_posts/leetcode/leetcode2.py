from ListNode import ListNode
class leetcode2:
    def addTwoNumber(self, l1:ListNode, l2:ListNode) -> ListNode:
        carry = 0
        ret = ListNode(0)
        cur = ret
        while (l1 != None or l2 != None):
            if l1 == None:
                val1 = 0
            else:
                val1 = l1.val
            if l2 == None:
                val2 = 0
            else:
                val2 = l2.val
            num = val1 + val2 + carry
            val = num % 10
            carry = num // 10
            ret.next = ListNode(val)
            ret = ret.next
            if (l1 == None):
                l1 = None
            else:
                l1 = l1.next
            if (l2 == None):
                l2 = None
            else:
                l2 = l2.next
        if carry == 1:
            ret.next = ListNode(1)
        return cur.next
if __name__ == '__main__':
#链表1
    print("\n")
    print("l1:")
    l1 = input()
    l1 = l1.split(",")
    l1 = [int(l1[i]) for i in range(len(l1))]
    ls1 = ListNode(0)
    cur = ls1
    for i in range(len(l1)):
        ls1.next = ListNode(l1[i])
        ls1 = ls1.next
    ls1 = cur.next
#链表2
    print("l2:")
    l2 = input()
    l2 = l2.split(",")
    l2 = [int(l2[i]) for i in range(len(l2))]
    ls2 = ListNode(0)
    cur = ls2
    for i in range(len(l2)):
        ls2.next = ListNode(l2[i])
        ls2 = ls2.next
    ls2 = cur.next
    leetcode2 = leetcode2()
    ret = leetcode2.addTwoNumber(ls1, ls2)
    #print("return:", ret)
    while(ret != None):
        print(str(ret.val) + "-->", end="")
        ret = ret.next