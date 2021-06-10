import java.util.*;
/*二叉树的右视图
输入: [1,2,3,null,5,null,4]
输出: [1, 3, 4]
解释:

   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---

* */
public class leetcode199 {
    public static void main(String[] args) {
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.left.right = new TreeNode(5);
        root.right.right = new TreeNode(4);
        List<Integer> ret = rightSideView(root);
        Iterator<Integer> it = ret.iterator();
        while(it.hasNext()){
            int temp = it.next();
            System.out.print(temp+"-->");
        }
    }
    public static List<Integer> rightSideView(TreeNode root){
        List<Integer> ret = new ArrayList<>();
        if(root==null){
            return ret;
        }
        Queue<TreeNode> que = new LinkedList<>();
        que.offer(root);
        while(!que.isEmpty()){
            int level_size = que.size();
            for(int i=0; i<level_size; i++){
                TreeNode temp = que.poll();
                if(i==level_size-1){
                    ret.add(temp.val);
                }
                if(temp.left!=null){
                    que.add(temp.left);
                }
                if(temp.right!=null){
                    que.add(temp.right);
                }
            }
        }
        return ret;
    }
}
