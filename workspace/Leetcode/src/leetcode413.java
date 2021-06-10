public class leetcode413 {
    public static void main(String[] args) {
        int[] nums = {1,2,3,4};
        //System.out.println(numberOfArithmeticSlices(nums));
        System.out.println(numberOfArithmeticSlices(nums));
    }
    /*public static int f(int[] nums){
        int count = 0;
        for(int i=0; i<nums.length-2; i++){
            int d = nums[i+1]-nums[i];
            for(int j=i+2; j<nums.length; j++){
                int k = 0;
                for(int k=i+1; k<=j; k++)
                    if(nums[k]-nums[k-1]!=d)
                        break;
                if(k>j)
                    count++;

            }
        }
        return count;
    }*/
    public static int numberOfArithmeticSlices(int[] A) {
        int count = 0;
        for (int s = 0; s < A.length - 2; s++) {
            int d = A[s + 1] - A[s];
            for (int e = s + 2; e < A.length; e++) {
                int i = 0;
                for (i = s + 1; i <= e; i++)
                    if (A[i] - A[i - 1] != d)
                        break;
                if (i > e)
                    count++;
            }
        }
        return count;
    }
}
