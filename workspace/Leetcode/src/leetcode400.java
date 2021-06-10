import org.omg.Messaging.SYNC_WITH_TRANSPORT;
/*leetcode 400 第N位数字
    在无限的整数序列 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...中找到第 n 位数字。
    输入11  对应   输出0
* */

public class leetcode400 {
    public static void main(String[] args) {
        int n = 1000000000;
        System.out.println(findNthDigit(n));
    }
    public static int f(long n){
        //1000000000无法通过,原因
        //base在乘法运算中，可能会溢出int范围
        long base = 9;
        int digt = 1;
        while(n-base*digt>0){
            n = n - base*digt;
            base = base*10;
            digt += 1;
        }
        long idx = n%digt;
        if(idx==0){
            idx = digt;
        }
        long number = 1;
        for(int i=1; i<digt; i++){
            number *= 10;
        }
        if(idx==digt){
            number += n/digt -1;
        }
        else{
            number += n/digt;
        }
        for(long i=idx; i<digt; i++){
            number = number/10;
        }
        return (int)number%10;
    }
    public static int findNthDigit(int n) {
        if (n < 10) return n;
        int index = 2;
        int sum = 9;
        while (sum + 9 * Math.pow(10, index - 1) * index < n) {
            sum += 9 * Math.pow(10, index - 1) * index;
            index++;
        }
        System.out.println("index: "+index);
        //在index位数里的位置
        int pos = n - sum;
        //获取在哪个数
        int i = (int) (Math.pow(10, index - 1) + (pos / index));
        System.out.println("i: "+i);
        //获取在数的第几位
        int i1 = pos % index;
        System.out.println("i1: "+i1);
        if (i1 != 0) {
            for (int j = 0; j < index - i1; j++) {
                i /= 10;
                System.out.println("for i: "+i);
            }
            return i % 10;
        } else {
            return (i - 1) % 10;
        }
    }
}
