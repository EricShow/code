import java.util.ArrayList;
import java.util.*;
//leetcode18 能够输出4个和为target的列表，但是没有考虑去重问题   输入[2,2,2,2,2]输出5个[2,2,2,2]，但实际我们只需要一个
/*      给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，
        使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。
        输入：nums = [1,0,-1,0,-2,2], target = 0
        输出：[[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
* */
//
public class leetcode18 {
    static List<List<Integer>> ret = new ArrayList<>();
    public static void main(String[] args) {
        int[] arr = {1,0,-1,0,-2,2};
        int target = 0;
        System.out.println(sumof(arr, target));
    }

    public static List<List<Integer>> sumof(int[] nums, int target){
        Arrays.sort(nums);
        List<List<Integer>> ret = new ArrayList<>();
        List<Integer> ls = new ArrayList<>();
        f(nums,target,0,4,ls,ret);
        return ret;
    }
    public static void f(int[] arr, int target, int start, int k, List<Integer> ls,List<List<Integer>> ret){
        if(k<0){
            return;
        }
        if(k==0&&target==0){
            //System.out.println(k);
            ret.add(new ArrayList<>(ls));
            return ;
        }
        for(int i=start; i<arr.length; i++){
            /*这种去重会不对，输出结果不够
            if(i>0&&arr[i] == arr[i-1]){
                continue;
            }
            */

            ls.add(arr[i]);
            f(arr,target-arr[i],i+1,k-1,ls,ret);
            ls.remove(ls.size()-1);
        }
    }
}