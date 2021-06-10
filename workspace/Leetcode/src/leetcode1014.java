/*
leetcode1014 最佳观光组合
输入：values = [8,1,5,2,6]
输出：11
解释：i = 0 j = 2 values[i]+values[j]+i-j = 8+5+0-2 = 11
* */
public class leetcode1014 {
    public static void main(String[] args) {
        int[] values = {8,1,5,2,6};
        System.out.println(maxScoreSightseeingPair(values));
    }
    public static int maxScoreSightseeingPair(int[] values){
        int n = values.length;
        int max = Integer.MIN_VALUE;
        //int[][] dp = new int[n][n];
        for(int i=0; i<n-1; i++){
            for(int j=i+1; j<n; j++){
                max = Math.max(values[j]+values[i]-j+i, max);
            }
        }
        return max;
    }
}
