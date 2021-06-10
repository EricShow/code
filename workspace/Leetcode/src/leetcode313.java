import java.util.*;
/*
* 超级丑数
    输入: n = 12, primes = [2,7,13,19]
    输出: 32
    解释: 给定长度为 4 的质数列表 primes = [2,7,13,19]，前 12 个超级丑数序列为：[1,2,4,7,8,13,14,16,19,26,28,32] 。

* */
public class leetcode313 {
    public static void main(String[] args) {
        int n = 12;
        int[] primes = {2,7,13,19};
        System.out.println(nthSuperUglyNumber(n, primes));
    }
    public static int nthSuperUglyNumber(int n, int[] primes){
        int len = primes.length;
        Map<Integer,Integer> hash = new HashMap<>();
        for(int i=0; i<len; i++){
            hash.put(primes[i],1);
        }

        int[] dp = new int[n+1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[1] = 1;
        for(int i=2; i<n+1; i++){
            for(int j=0; j<len; j++){
                int num = dp[hash.get(primes[j])] * primes[j];
                dp[i] = Math.min(num,dp[i]);

            }
            for(int j=0; j<len; j++){
                if(dp[i] == dp[hash.get(primes[j])] * primes[j]){
                    //System.out.println("here"+hash.get(primes[j]));
                    hash.put(primes[j],hash.get(primes[j])+1);
                }
            }
        }

        return dp[n];
    }
}
