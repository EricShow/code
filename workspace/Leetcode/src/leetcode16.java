import java.util.*;
//leetcode16 最接近的三数之和
//
/*  给定一个包括 n 个整数的数组 nums 和 一个目标值 target。找出 nums 中的三个整数，
    使得它们的和与 target 最接近。返回这三个数的和。假定每组输入只存在唯一答案。
    输入：nums = [-1,2,1,-4], target = 1
    输出：2
    解释：与 target 最接近的和是 2 (-1 + 2 + 1 = 2) 。
* */
public class leetcode16 {
    public static void main(String[] args) {
        int[] arr = {-1,2,1,-4};
        Arrays.sort(arr);
        int target = 2;
        int k = 3;
        //System.out.println(f(arr,k,Integer.MAX_VALUE,0,target));
        System.out.println(f1(arr,target));
    }
    public static int f(int[] arr, int k, int min, int index, int target){
        if(k==0 && Math.abs(target)<Math.abs(min)){
            System.out.println("执行if");
            min = target;
            System.out.println(k);
            System.out.println(min);
        }
        for(int i=index; k!=0&&i<arr.length; i++){
            System.out.println("执行for");
            target = target - arr[i];
            System.out.println(target);
            f(arr,k--,min,index+1,target);
            target = target + arr[i];
        }
        return min;
    }
    public static int f1(int[] arr,int target){
        Arrays.sort(arr);
        int ans = arr[0]+arr[1]+arr[2];
        int min = ans;
        for(int i=0; i<arr.length-2;i++){
            int start = i+1;
            int end = arr.length-1;
            while(start<end){
                int sum = arr[start]+arr[end]+arr[i];
                if(Math.abs(sum-target)<Math.abs(min-target)){
                    min = sum;
                }
                if(sum>target){
                    end--;
                }else if(sum<target){
                    start++;
                }else{
                    return min;
                }
            }
        }
        return min;


    }
}