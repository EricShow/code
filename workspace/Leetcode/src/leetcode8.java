import java.security.PublicKey;
import java.util.*;
//import java.util.*;
//力扣8 字符串转整数
/*      输入：s = "42"
        输出：42
        解释：加粗的字符串为已经读入的字符，插入符号是当前读取的字符。
        第 1 步："42"（当前没有读入字符，因为没有前导空格）
                 ^
        第 2 步："42"（当前没有读入字符，因为这里不存在 '-' 或者 '+'）
                 ^
        第 3 步："42"（读入 "42"）
                   ^
        解析得到整数 42 。
        由于 "42" 在范围 [-231, 231 - 1] 内，最终结果为 42 。
* */
public class leetcode8 {
    public static void main(String[] args) {
        String s = "  4193 with words";
        System.out.println(myAtoi(s));
        System.out.println(f(s));
    }
    public static int f(String s){
        String str = s.trim();
        System.out.println(str);
        //char[] ch = str.toCharArray();
        int length = str.length();
        System.out.println(length);
        int flag = 1;
        int start = 1;
        if(str.charAt(0)=='-'){
            flag = -1;
        }else if(str.charAt(0)=='+'){
            flag = 1;
        }else if(str.charAt(0)>='0'&&str.charAt(0)<='9'){
            start = 0;;
        }
        int sum = 0;
        for(int i=start; i<length; i++){
            if (sum > Integer.MAX_VALUE / 10 || (sum == Integer.MAX_VALUE / 10 && (str.charAt(i) - '0') > Integer.MAX_VALUE % 10)) {
                return Integer.MAX_VALUE;
            }
            if (sum < Integer.MIN_VALUE / 10 || (sum == Integer.MIN_VALUE / 10 && (str.charAt(i) - '0') > -(Integer.MIN_VALUE % 10))) {
                return Integer.MIN_VALUE;
            }
            System.out.println(str.charAt(i));
            if(str.charAt(i)>='0'&&str.charAt(i)<='9'){
                sum = sum*10+str.charAt(i)-'0';
            }else{
                return sum;
            }
        }
        return sum*flag;
    }
    public static  int myAtoi(String str) {
        int len = str.length();
        // str.charAt(i) 方法回去检查下标的合法性，一般先转换成字符数组
        char[] charArray = str.toCharArray();

        // 1、去除前导空格
        int index = 0;
        while (index < len && charArray[index] == ' ') {
            index++;
        }

        // 2、如果已经遍历完成（针对极端用例 "      "）
        if (index == len) {
            return 0;
        }

        // 3、如果出现符号字符，仅第 1 个有效，并记录正负
        int sign = 1;
        char firstChar = charArray[index];
        if (firstChar == '+') {
            index++;
        } else if (firstChar == '-') {
            index++;
            sign = -1;
        }

        // 4、将后续出现的数字字符进行转换
        // 不能使用 long 类型，这是题目说的
        int res = 0;
        while (index < len) {
            char currChar = charArray[index];
            // 4.1 先判断不合法的情况
            if (currChar > '9' || currChar < '0') {
                break;
            }

            // 题目中说：环境只能存储 32 位大小的有符号整数，因此，需要提前判：断乘以 10 以后是否越界
            if (res > Integer.MAX_VALUE / 10 || (res == Integer.MAX_VALUE / 10 && (currChar - '0') > Integer.MAX_VALUE % 10)) {
                return Integer.MAX_VALUE;
            }
            if (res < Integer.MIN_VALUE / 10 || (res == Integer.MIN_VALUE / 10 && (currChar - '0') > -(Integer.MIN_VALUE % 10))) {
                return Integer.MIN_VALUE;
            }

            // 4.2 合法的情况下，才考虑转换，每一步都把符号位乘进去
            res = res * 10 + sign * (currChar - '0');
            index++;
        }
        return res;
    }

}