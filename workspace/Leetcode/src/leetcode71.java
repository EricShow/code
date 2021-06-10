import java.util.Deque;
import java.util.LinkedList;

public class leetcode71 {
    public static void main(String[] args) {
        String path = "/../";
        System.out.println(simlifyPath(path));
    }
    public static String simlifyPath(String path){
        Deque<String> queue = new LinkedList<>();
        String[] res = path.split("/");
        for(int i=0; i<res.length;i++){
            String s = res[i];
            if(s.equals(".")||s.equals("")){
                continue;
            }
            else if(s.equals("..")){
                if(s.isEmpty()){
                    queue.poll();
                }
            }else{
                queue.offer(s);
            }

        }
        StringBuffer sb = new StringBuffer("/");
        while(!queue.isEmpty()){
            sb.append(queue.poll());
            if(!queue.isEmpty()){
                sb.append("/");
            }
        }
        return sb.toString().equals("")?"/":sb.toString();
    }
}
