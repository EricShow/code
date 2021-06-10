import java.util.Scanner;
/*长度最小的子数组
输入：target = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。
* */
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