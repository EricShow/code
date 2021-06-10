import java.util.*;
/*只出现一次的数字
输入：nums = [1,2,1,3,2,5]
输出：[3,5]
解释：[5, 3] 也是有效的答案。
* */
public class leetcode260 {
    public static void main(String[] args) {
        int[] nums = {1,2,1,3,5,2};
        int[] ret = f(nums);
        System.out.println(ret[0]+"  "+ret[1]);
    }
    public static int[] f(int[] nums){
        Map<Integer,Integer> hash = new HashMap<>();
        for(int i=0; i<nums.length; i++){
            hash.put(nums[i],hash.getOrDefault(nums[i],0)+1);
        }
        int count = 0;
        int[] ret = new int[2];
        for(int i=0; i<nums.length; i++){
            if(hash.get(nums[i])==1&&count<2){
                ret[count] = nums[i];
                count++;
                if(count==2){
                    return ret;
                }
            }else{
                continue;
            }
        }
        return ret;
    }
}
