import java.util.Arrays;
/*H指数
*
*   输入：citations = [3,0,6,1,5]
    输出：3
解释：给定数组表示研究者总共有 5 篇论文，每篇论文相应的被引用了 3, 0, 6, 1, 5 次。
     由于研究者有 3 篇论文每篇 至少 被引用了 3 次，其余两篇论文每篇被引用 不多于 3 次，所以她的 h 指数是 3。
。*/
public class leetcode274 {
    public static void main(String[] args) {
        int[] citations = {3,0,6,1,5};
        System.out.println(hIndex(citations));
    }
    public static int hIndex(int[] citations){
        Arrays.sort(citations);//从小到大排序
        for(int i=0; i<citations.length; i++){
            System.out.print(citations[i]+", ");
        }
        System.out.println("");
        // 线性扫描找出最大的 i
        int i = 0;
        while (i < citations.length && citations[citations.length - 1 - i] > i) {
            //索引为citations.length-1-i的值大于1，说明该索引后面都大于1
            System.out.println("herehere");
            i++;
        }
        return i;
    }
}
