public class leetcode477 {
    public static void main(String[] args) {
        int[] nums = {4,14,2};
        System.out.println(totalHammingDistance(nums));
    }
    public static int totalHammingDistance(int[] nums){
        int ret = 0;
        for(int i=0; i<nums.length; i++){
            for(int j=i+1; j<nums.length; j++){
                ret += HammingDistance(nums[i],nums[j]);
            }
        }
        return ret;
    }
    public static int HammingDistance(int a, int b){
        return Integer.bitCount(a^b);
    }
}

