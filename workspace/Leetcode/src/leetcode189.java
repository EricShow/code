import org.omg.Messaging.SYNC_WITH_TRANSPORT;
/*旋转数组
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右旋转 1 步: [7,1,2,3,4,5,6]
向右旋转 2 步: [6,7,1,2,3,4,5]
向右旋转 3 步: [5,6,7,1,2,3,4]

* */
public class leetcode189 {
    public static void main(String[] args) {
        int[] nums = {1,2,3,4,5,6,7};
        int k = 3;
        int[] ret = rotate(nums,k);
        for(int i=0; i<nums.length; i++) {
            System.out.println(ret[i]);
        }
    }
    public static int[] rotate(int[] nums, int k){
        k = k%nums.length;
        int[] arr = new int[k];
        for(int i=nums.length-k; i<nums.length; i++){
            //System.out.println(nums[i]);
            arr[i-nums.length+k] = nums[i];
        }
        int[] temp = new int[nums.length-k];
        for(int i=nums.length-k-1; i>=0; i--){
            nums[i+k] = nums[i];
        }
        for(int i=0; i<k; i++){
            nums[i] = arr[i];
        }
        return nums;
    }
}
