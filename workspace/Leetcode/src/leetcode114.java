import java.util.LinkedList;
import java.util.*;
/*  给你二叉树的根结点 root ，请你将它展开为一个单链表：
    展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
    展开后的单链表应该与二叉树 先序遍历 顺序相同。
*/
public class leetcode114 {
    public static void main(String[] args) {
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.left.right = new TreeNode(5);
        root.right.right = new TreeNode(4);
        TreeNode ret = flatten(root);
        while(ret!=null){
            int temp = ret.val;
            ret = ret.right;
            System.out.print(temp+"-->");
        }
    }


    public static TreeNode flatten(TreeNode root){
        List<TreeNode> list = new ArrayList<TreeNode>();
        dfs(root, list);
        int size = list.size();
        for (int i = 1; i < size; i++) {
            TreeNode prev = list.get(i - 1), curr = list.get(i);
            prev.left = null;
            prev.right = curr;
        }
        return root;
    }
    public static void dfs(TreeNode root, List<TreeNode> ret){
        if(root==null){
            return;
        }
        ret.add(root);
        dfs(root.left, ret);
        dfs(root.right, ret);
    }
}
