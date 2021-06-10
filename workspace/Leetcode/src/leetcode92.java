import java.util.*;
public class leetcode92 {
/* 反转链表Ⅱ
*   输入：head = [1,2,3,4,5], left = 2, right = 4
    输出：[1,4,3,2,5]
* */
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        ListNode head = new ListNode(0);
        List<Integer> arr = new ArrayList<>();
        while(sc.hasNext()){
            int temp = sc.nextInt();
            if(temp==999){
                break;
            }
            arr.add(temp);
        }
        ListNode cur = new ListNode(0);
        cur.next = head;
        System.out.println("arr.size: "+arr.size());
        for(int i=0; i<arr.size()&&head!=null; i++){
            ListNode ans = new ListNode(arr.get(i));
            head.next = ans;
            head = head.next;
        }
        System.out.println("left: ");
        int left = sc.nextInt();
        System.out.println("right: ");
        int right = sc.nextInt();
        System.out.println("cur.next: "+ cur.next.val);
        ListNode ret = reverseBetween(cur.next, left, right);
        while(ret!=null){
            System.out.print(ret.val+"-->");
            ret = ret.next;
        }
    }
    public static ListNode reverseBetween(ListNode head, int left, int right){
        ListNode leftNode = head;
        ListNode rightNode = head;
        Stack<ListNode> stack = new Stack<>();

        for(int i=0; i<(left+right)/2; i++){
            rightNode = rightNode.next;

        }
        for(int i=(left+right)/2+1; i<=right; i++){
            stack.push(rightNode);
            rightNode = rightNode.next;
        }
        for(int i=0; i<left-1; i++){
            leftNode = leftNode.next;
        }
        while(!stack.isEmpty()){
            rightNode = stack.pop();
            int temp = rightNode.val;
            rightNode.val = leftNode.val;
            leftNode.val = temp;
            leftNode = leftNode.next;
        }
        return head;
    }
}