import java.util.*;
/*
leetcode516. 最长回文子序列
输入："bbbab"
输出：4
* */
public class leetcode516 {
    public static void main(String[] args) {
        String s = "bbbab";
        System.out.println(longestPalindromeSubseq(s));
    }
    public static int longestPalindromeSubseq(String s){
        char[] ch = s.toCharArray();
        int n = ch.length;
        int[][] f = new int[n][n];
        for(int i=ch.length-1; i>=0; i--){
            f[i][i] = 1;
            for(int j=i+1; j<ch.length; j++){
                if(ch[i]==ch[j]) {
                    f[i][j] = f[i + 1][j - 1] + 2;
                }else{
                    f[i][j] = Math.max(f[i+1][j],f[i][j-1]);
                }
            }
        }
        return f[0][n-1];
    }
}
