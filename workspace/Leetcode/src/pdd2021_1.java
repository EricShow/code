import java.util.Scanner;

public class pdd2021_1 {
    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int num = in.nextInt();
        System.out.println(f(num));
    }
    public static int f(int num){
        int ans = 0, max = 9, level = 1;
        if(num > 45){
            ans = -1;
        }
        else{
            while(num > 0){
                if(num > max){
                    ans += max * level;
                    level *= 10;
                }else ans += num *level;
                num -= max;
                max--;
            }
            //System.out.println(ans);
        }
        return ans;
    }
}
