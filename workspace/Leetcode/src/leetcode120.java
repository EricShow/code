import java.util.*;
public class leetcode120 {
    public static void main(String[] args) {

        List<Integer> in1 = new ArrayList<>();
        in1.add(-1);
        List<Integer> in2 = new ArrayList<>();
        in2.add(-2);
        in2.add(-3);

        /*List<Integer> in1 = new ArrayList<>();
        in1.add(2);
        List<Integer> in2 = new ArrayList<>();
        in2.add(3);
        in2.add(4);
        List<Integer> in3 = new ArrayList<>();
        in3.add(6);
        in3.add(5);
        in3.add(7);
        List<Integer> in4 = new ArrayList<>();
        in4.add(4);
        in4.add(1);
        in4.add(8);
        in4.add(3);*/

        List<List<Integer>> in = new ArrayList<>();
        in.add(in1);
        in.add(in2);

        System.out.println(minimumTotal_sdl(in));
    }
    public static int minimumTotal_sdl(List<List<Integer>> triangle){
        if(triangle.size()==1){
            return triangle.get(0).get(0);
        }
        int len = triangle.size();
        List<List<Integer>> ret = new ArrayList<>();
        for(int i=0; i<len; i++){
            List<Integer> ls = new ArrayList<>();
            if(i==0){
                ls.add(triangle.get(i).get(0));
                ret.add(ls);
                continue;
            }
            else{
                for(int j=0; j<triangle.get(i).size(); j++){
                    if(j==0){
                        ls.add(ret.get(i-1).get(j)+triangle.get(i).get(j));
                    }else if(j==(triangle.get(i).size()-1)){
                        ls.add(ret.get(i-1).get((triangle.get(i-1).size()-1))+triangle.get(i).get((triangle.get(i).size()-1)));
                    }else{
                        int tmp1 = ret.get(i-1).get(j)+triangle.get(i).get(j);
                        int tmp2 = ret.get(i-1).get(j-1)+triangle.get(i).get(j);
                        ls.add(Math.min(tmp1,tmp2));
                    }
                }
                ret.add(ls);
            }
        }
        int min = Integer.MAX_VALUE;
        for(int i=0; i<triangle.get(triangle.size()-1).size(); i++){
            min = Math.min(ret.get(triangle.size()-1).get(i), min);
        }
        return min;
    }
    public static int minimumTotal(List<List<Integer>> triangle){
        int n = triangle.size();
        int[][] f = new int[n][n];
        f[0][0] = triangle.get(0).get(0);
        for (int i = 1; i < n; ++i) {
            f[i][0] = f[i - 1][0] + triangle.get(i).get(0);
            for (int j = 1; j < i; ++j) {
                f[i][j] = Math.min(f[i - 1][j - 1], f[i - 1][j]) + triangle.get(i).get(j);
            }
            f[i][i] = f[i - 1][i - 1] + triangle.get(i).get(i);
        }
        int minTotal = f[n - 1][0];
        for (int i = 1; i < n; ++i) {
            minTotal = Math.min(minTotal, f[n - 1][i]);
        }
        return minTotal;
    }
}
