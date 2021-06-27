### Leetcode90

![image-20210518214720468](D:\Markdown\images\image-20210518214720468.png)

****

**回溯法基本思想**		

​		首先排序，排序是为了方便去重
​		相同的元素在同一层中是不允许使用的	
​		相同元素在同一路径中是可以使用的

```java

import java.util.*;
public class leetcode90 {
    private static List<List<Integer>> ans;
    private static List<Integer> path;

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int[] nums = new int[n];
        for(int i=0; i<nums.length; i++){
            nums[i] = sc.nextInt();
        }
        List<List<Integer>> ret = subsetsWithDup(nums);
        Iterator<List<Integer>> it = ret.iterator();
        while(it.hasNext()){
            List<Integer> temp = it.next();
            System.out.println("value: "+temp);
        }

    }

    public static List<List<Integer>> subsetsWithDup(int[] nums) {
        ans = new ArrayList<>();
        path = new ArrayList<>();
        // 首先排序，让相同的两个元素排到一起去，便于去重
        Arrays.sort(nums);
        int n = nums.length;
        // 使用 visited 数组来记录哪一个元素在当前路径中被使用了
        boolean[] visited = new boolean[n];
        // 开始回溯
        backtrace(nums, 0, visited, n);
        return ans;
    }

    private static void backtrace(int[] nums, int start, boolean[] visited, int n) {
        // 首先加入当前路径
        ans.add(new ArrayList<>(path));
        // 从 start 开始遍历每一个元素，尝试加入路径中
        for (int i = start; i < n; ++i) {
            // 如果当前元素和前一个元素相同，而且前一个元素没有被访问，说明前一个相同的元素在当前层已经被用过了
            if (i > 0 && nums[i - 1] == nums[i] && !visited[i - 1]) continue;
            // 记录下来，用过了当前的元素
            visited[i] = true;
            path.add(nums[i]); // 放到路径中
            backtrace(nums, i + 1, visited, n); // 向下一个递归
            visited[i] = false; // 回溯
            path.remove(path.size() - 1);
        }
    }
}
```

### Leetcode209

**方法一：暴力法**

暴力法是最直观的方法，初始化子数组的最小长度为无穷大，枚举数组nums中每个下标作为子数组的开始下标，对于每个开始下标i，需要找到大于或等于i的最小下标j，使得从nums[i]到nums[j]的元素和大于或等于s，并更新子数组的最小长度（此时子数组的长度是j-i+1）

![image-20210518225007862](D:\Markdown\images\image-20210518225007862.png)

```java
import java.util.Scanner;

public class leetcode209 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("输入目标值s=: ");
        int s = sc.nextInt();
        System.out.println("输入数组长度n=: ");
        int n = sc.nextInt();
        int[] nums = new int[n];
        for(int i=0; i<n; i++){
            nums[i] = sc.nextInt();
            //System.out.println(nums[i]);
        }
        //System.out.println(nums[3]);s
        System.out.println(nums.length);
        for(int j=0; j<nums.length; j++){
            
            System.out.println("nums["+j+"] = "+nums[j]);
        } 
        
        System.out.println(minSubArray(s, nums));
    }
    public static int minSubArray(int s, int[] nums){
        int min = Integer.MAX_VALUE;
        for(int i=0; i<nums.length; i++){
            int sum = nums[i];
            if(sum>=s){
                return 1;
            }
            for(int j=i+1; j<nums.length; j++){
                sum += nums[j];
                if(sum>=s){
                    min = Math.min(min,j-i+1);
                    break;
                }
            }
        }
        return min==Integer.MAX_VALUE?0:min;

    }
}
```

### Leetcode92



![image-20210518234951656](D:\Markdown\images\image-20210518234951656.png)

**解题思路**

用在数组中反转区间值的思路，可以将[left, rught]中的元素看成左右两部分，left ~ (left + right) / 2 - 1, (left + right) / 2 + 1 ~ right。前半部分从前往后遍历，后半部分从后往前遍历，边遍历边交换val，就可以实现反转链表。然而，我们需要记录后半部分每个元素的前驱结点，才能实现从后往前遍历。又因为节点之前的前后关系，可以使用stack，首先从前往后便后半部分，并将节点元素压入栈内。
最后用一个循环找到left节点，就可以开始从前往后遍历前半部分和从后往前遍历后半部分了

```java
import java.util.*;
public class leetcode92 {
    /* 反转链表Ⅱ */
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
```

