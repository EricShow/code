import sun.reflect.generics.tree.Tree;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
/*二叉树的中序遍历
* */
public class leetcode94 {
    public static void main(String[] args) {
        TreeNode root = new TreeNode(1);
        root.left = new TreeNode(2);
        root.right = new TreeNode(3);
        root.left.left = new TreeNode(4);
        root.right.left = new TreeNode(5);
        root.right.right = new TreeNode(6);
        root.right.right.left = new TreeNode(7);
        List<Integer> ret = inorderTraversal(root);
        Iterator<Integer> it = ret.iterator();
        while(it.hasNext()){
            int temp = it.next();
            System.out.print(temp+"-->");
        }
        //output：4-->2-->1-->5-->3-->7-->6-->
    }
    public static List<Integer> inorderTraversal(TreeNode root){
        List<Integer> res = new ArrayList<>();
        inorder(root,res);
        return res;
    }
    public static void inorder(TreeNode root, List<Integer> res){
        if(root==null){
            return;
        }
        inorder(root.left,res);
        res.add(root.val);
        inorder(root.right,res);
    }
}
