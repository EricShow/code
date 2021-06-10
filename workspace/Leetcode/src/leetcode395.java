import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
/*题目：至少有k个重复字符的最长子串
输入：s = "aaabb", k = 3
输出：3
解释：最长子串为 "aaa" ，其中 'a' 重复了 3 次。
* */
public class leetcode395 {
    public static void main(String[] args) {
        String s = "ababbc";
        int k = 2;
        System.out.println(longestSubstring(s, k));
    }
    public static int longestSubstring(String s, int k) {
        Map<Character,Integer> hash = new HashMap<>();
        char[] ch = s.toCharArray();
        for(int i=0; i<ch.length; i++){
            hash.put(ch[i], hash.getOrDefault(ch[i],0)+1);
        }
        List<Character> ls = new ArrayList<>();
        int max = 0;
        for(int i=0; i<ch.length; i++){
            if(hash.get(ch[i])<k){
                //System.out.println("aaaaaaaaaaaaaaa");
                ls = new ArrayList<>();
                int j = 0;
                while(ls.contains(ch[i])){
                    //
                    ls.remove(j);
                    j++;
                }
            }else{
                //System.out.println("bbbbbbbb");
                ls.add(ch[i]);
            }
            max = Math.max(ls.size(),max);
        }
        //System.out.println(ls);
        return max;
    }
}
