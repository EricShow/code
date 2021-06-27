import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;

public class TestSplit {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        String input = sc.nextLine();
        String [] s = input.trim().split("\\s+");

//键盘输入链表 用空格隔开  1 2 3 4 56   7
        ListNode ls = new ListNode(0);
        ListNode cur = ls;
        for(int i=0; i<s.length; i++){
            ListNode tmp = new ListNode(Integer.valueOf(s[i]));
            cur.next = tmp;
            cur = cur.next;
        }
        //Iterator<Integer> it = arr.iterator();
        ls = ls.next;
        while(ls!=null){
            System.out.print(ls.val+"-->");
            ls = ls.next;
        }
        System.out.println(" ");

//键盘输入列表 用空格隔开  1 2 3 4 56   7
        List<Integer> arr = new ArrayList<>();
        for(int i=0; i<s.length; i++){
            arr.add(Integer.valueOf(s[i]));
        }
        Iterator<Integer> it = arr.iterator();
        while(it.hasNext()){
            int tmp = it.next();
            System.out.print(tmp+"-->");
        }
    }
    public static int f(ListNode a){
        return a.val;
    }
}
