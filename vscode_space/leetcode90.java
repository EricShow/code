import java.util.*;
public class leetcode90 {
    private static List<List<Integer>> ans;
    private static List<Integer> path;

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int[] nums = new int[n];
        for(int i=0; i<nums.length; i++){
            nums[i] = sc.nextInt();
        }
        List<List<Integer>> ret = subsets(nums);
        Iterator<List<Integer>> it = ret.iterator();
        while(it.hasNext()){
            List<Integer> temp = it.next();
            System.out.println("value: "+temp);
        }

    }
    public static List<List<Integer>> subsets(int[] nums){
        ans = new ArrayList<>();
        path = new ArrayList<>();
        Arrays.sort(nums);
        int len = nums.length;
        boolean[] visited = new boolean[len];
        f(nums, 0, visited, len);
        return ans;
    }
    public static void f(int[] nums, int start, boolean[] visited, int len){
        ans.add(new ArrayList<>(path));
        for(int i=start; i<len; i++){
            if(i>0&&nums[i]==nums[i-1]&&visited[i]==false){
                continue;
            }
            visited[i]=true;
            path.add(nums[i]);
            f(nums,i+1,visited,len);
            visited[i] = false; // 回溯
            path.remove(path.size() - 1);
        }
    }






























    public static List<List<Integer>> subsetsWithDup(int[] nums) {
        ans = new ArrayList<>();
        path = new ArrayList<>();
        // 首先排序，让相同的两个元素排到一起去，便于去重
        Arrays.sort(nums);
        int n = nums.length;
        // 使用 visited 数组来记录哪一个元素在当前路径中被使用了
        boolean[] visited = new boolean[n];
        // 开始回溯
        backtrace(nums, 0, visited, n);
        return ans;
    }

    private static void backtrace(int[] nums, int start, boolean[] visited, int n) {
        // 首先加入当前路径
        ans.add(new ArrayList<>(path));
        // 从 start 开始遍历每一个元素，尝试加入路径中
        for (int i = start; i < n; ++i) {
            // 如果当前元素和前一个元素相同，而且前一个元素没有被访问，说明前一个相同的元素在当前层已经被用过了
            if (i > 0 && nums[i - 1] == nums[i] && !visited[i - 1]) continue;
            // 记录下来，用过了当前的元素
            visited[i] = true;
            path.add(nums[i]); // 放到路径中
            backtrace(nums, i + 1, visited, n); // 向下一个递归
            visited[i] = false; // 回溯
            path.remove(path.size() - 1);
        }
    }
}