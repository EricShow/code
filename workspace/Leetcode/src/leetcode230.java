import java.util.ArrayList;
import java.util.List;
/*二叉搜索树中第k小的元素
给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 个最小元素（从 1 开始计数）。

* */
public class leetcode230 {
    public static void main(String[] args) {
        TreeNode root = new TreeNode(3);
        root.left = new TreeNode(1);
        root.right = new TreeNode(4);
        root.left.right = new TreeNode(2);
        int k = 1;
        System.out.println(f(root,k));
    }
    public static int f(TreeNode root, int k){
        List<Integer> res = new ArrayList<>();
        dfs(root,res);
        return res.get(k-1);
    }
    public static void dfs(TreeNode root, List<Integer> res){
        if(root==null){
            return;
        }
        dfs(root.left,res);
        res.add(root.val);
        dfs(root.right,res);
    }
}
