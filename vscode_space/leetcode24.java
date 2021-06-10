import java.util.*;

public class leetcode24 {
    public static void main(String[] args) {
        ListNode head = new ListNode(1);
        head.next = new ListNode(2);
        head.next.next = new ListNode(3);
        head.next.next.next = new ListNode(4);
        //System.out.println(swapPairs(head));这样输出是地址
        ListNode a = swapPairs(head);
        while(a!=null){
            System.out.println(a.val);
            a = a.next;
        }
    }
    public static ListNode swapPairs(ListNode head){
        ListNode pre = new ListNode(0);
        pre.next = head;
        ListNode temp = pre;
        while(temp.next!=null&&temp.next.next!=null){
            ListNode start = temp.next;
            ListNode end = temp.next.next;
            temp.next = end;
            start.next = end.next;
            end.next = start;
            temp = start;
        }
        return pre.next;
    }
}