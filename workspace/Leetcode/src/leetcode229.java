import java.util.*;
/*求众数
输入：[3,2,3]
输出：[3]
* */
public class leetcode229 {
    public static void main(String[] args) {
        int[] nums = {1,1,1,3,3,2,2,2,1,2};
        List<Integer> ret = majorityElement(nums);
        Iterator<Integer> it = ret.iterator();
        while(it.hasNext()){
            int temp = it.next();
            System.out.println(temp);
        }
    }
    public static List<Integer> majorityElement(int[] nums){
        Map<Integer,Integer> hash = new HashMap<>();
        List<Integer> ls = new ArrayList<>();
        Set<Integer> set = new HashSet<>();
        for(int i=0; i<nums.length; i++){
            hash.put(nums[i],hash.getOrDefault(nums[i],0)+1);
            if(hash.get(nums[i])>(nums.length/3)&&!set.contains(nums[i])) {
                set.add(nums[i]);
                ls.add(nums[i]);
            }else{
                continue;
            }
        }
        return ls;
    }
}
