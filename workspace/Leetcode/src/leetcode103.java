import java.awt.*;
import java.util.*;
import java.util.List;

/*锯齿遍历：
给定一个二叉树，返回其节点值的锯齿形层序遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。
* */
public class leetcode103 {
    public static void main(String[] args) {
        TreeNode root = new TreeNode(3);
        root.left = new TreeNode(9);
        root.right = new TreeNode(20);
        root.right.left = new TreeNode(15);
        root.right.right = new TreeNode(7);
        List<List<Integer>> ret = f(root);
        Iterator<List<Integer>> it = ret.iterator();
        while(it.hasNext()){
            List<Integer> ans = it.next();
            for(int i=0; i<ans.size(); i++){
                System.out.print(ans.get(i)+"-->");
            }
            System.out.println("");
        }
    }
    public static List<List<Integer>> f(TreeNode root){
        List<List<Integer>> ret = new ArrayList<>();
        int level = 1;
        Deque<TreeNode> que = new ArrayDeque<>();
        Deque<TreeNode> stack = new ArrayDeque<>();
        que.offer(root);
        boolean flag = true;
        while(!que.isEmpty()||!stack.isEmpty()){
            if(level%2==1){
                flag = true;
            }else{
                flag = false;
            }
            List<Integer> ls = new ArrayList<>();
            if(flag){
                int que_size = que.size();
                for(int i=0; i<que_size; i++){
                    TreeNode tmp = que.poll();
                    ls.add(tmp.val);
                    if(tmp.left!=null){
                        stack.push(tmp.left);
                    }
                    if(tmp.right!=null){
                        stack.push(tmp.right);
                    }
                }
            }else{
                int stack_size = stack.size();
                for(int i=0; i<stack_size; i++){
                    TreeNode tmp = stack.poll();
                    ls.add(tmp.val);
                    if(tmp.right!=null){
                        que.addFirst(tmp.right);
                    }
                    if(tmp.left!=null){
                        que.addFirst(tmp.left);
                    }
                }
            }
            level++;
            ret.add(ls);
        }
        return ret;
    }
    public static List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> ret = new ArrayList<>();
        if(root==null){
            return ret;
        }
        Deque<TreeNode> queue = new ArrayDeque<>();
        Deque<TreeNode> stack = new ArrayDeque<>();

        queue.add(root);
        int level =1;///层数为奇数时，正常顺序遍历,层数为偶数,倒序遍历
        while (!queue.isEmpty()||!stack.isEmpty()){

            boolean isOrder = (level&1)==1;
            int n =isOrder?queue.size():stack.size();
            List<Integer> sub = new ArrayList<>();
            for(int i=0;i<n;i++){
                if(isOrder){
                    TreeNode curNode = queue.poll();
                    sub.add(curNode.val);
                    if(curNode.left!=null){
                        stack.push(curNode.left);
                    }
                    if(curNode.right!=null){
                        stack.push(curNode.right);
                    }
                }else{
                    TreeNode curNode = stack.poll();
                    sub.add(curNode.val);
                    if(curNode.right!=null){
                        queue.addFirst(curNode.right);
                    }
                    if(curNode.left!=null){
                        queue.addFirst(curNode.left);
                    }

                }
            }
            level++;
            ret.add(sub);
        }
        return ret;
    }
}
