/*丑数II
输入：n = 10
输出：12
解释：[1, 2, 3, 4, 5, 6, 8, 9, 10, 12] 是由前 10 个丑数组成的序列。
* */
public class leetcode264 {
    public static void main(String[] args) {
        int n = 10;
        System.out.println(nthUglyNumber(n));
    }
    public static int nthUglyNumber(int n){
        int p2 = 1, p3 = 1, p5 = 1;
        int[] dp =new int[n+1];
        dp[1] = 1;
        for(int i=2; i<=n; i++){
            int num2 = dp[p2] * 2, num3 = dp[p3] * 3, num5 = dp[p5] * 5;
            dp[i] = Math.min(Math.min(num2,num3),num5);
            if (dp[i] == num2) {
                p2++;
            }
            if (dp[i] == num3) {
                p3++;
            }
            if (dp[i] == num5) {
                p5++;
            }
        }
        return dp[n];
    }
}
