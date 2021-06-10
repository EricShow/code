public class leetcode371 {
    //两加数之和，不适用+、-符号
    /*  输入: a = 1, b = 2
        输出: 3
    */
    public static void main(String[] args) {
        int a = -2;
        int b = 3;
        System.out.println(getSum(a,b));
    }
    public static int  getSum(int a, int b){
        while(b!=0){
            System.out.print("a: "+a+"  b: "+b);
            System.out.println("");
            int temp = a^b;
            b = (a&b)<<1;
            a = temp;
        }
        return a;
    }
}
