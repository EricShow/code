import jdk.nashorn.internal.runtime.linker.LinkerCallSite;

import java.util.Iterator;
import java.util.List;

//旋转链表Ⅱ：
//1、将链表变成环状，并记录链表长度
//2、利用长度和k步确定需要移动的步数
//3、利用来l2从head开始确定环状拆解位置，l2.next为新head，l2.next=null
public class leetcode61 {
    public static void main(String[] args) {
        ListNode head = new ListNode(1);
        ListNode pre = new ListNode(0);
        pre.next = head;
        for(int i=2; i<10; i++) {
            ListNode ls = new ListNode(i);
            head.next = ls;
            head = head.next;
        }
        int k = 4;
        ListNode ret = rotateRight(pre.next, k);
        //ListNode ret = f(pre.next, k);
        while(ret != null){
            System.out.print(ret.val+"-->");
            ret = ret.next;
        }
    }
    public static ListNode f(ListNode head, int k){
        if(head==null||k==0){
            return head;
        }
        ListNode l1 = head;
        int len = 1;
        while(l1.next!=null){
            l1 = l1.next;
            len++;
        }
        l1.next = head;
        ListNode l2 = head;
        for(int i=1; i<len-k%len; i++){
            l2 = l2.next;
        }
        ListNode newHead = l2.next;
        l2.next = null;
        return newHead;

    }




















    public static ListNode rotateRight(ListNode head, int k) {
        if(head == null||k==0){
            return head;
        }
        //空的时候、或者k=0返回head
        ListNode l1 = head;
        int len = 1;
        //l1指向head的末尾
        while(l1.next != null){
            l1 = l1.next;
            len++;
        }
        //将链表变成环形
        l1.next = head;
        ListNode l2 = head;
        //l2指向head, 将l2移动到断开的地方，len-k%len;
        for(int i=1; i<len-k%len; i++){
            l2 = l2.next;
        }
        ListNode newHead = l2.next;
        l2.next = null;
        return newHead;
    }
}
