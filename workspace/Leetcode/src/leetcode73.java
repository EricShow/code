import java.util.*;
public class leetcode73 {
    public static void main(String[] args) {
        int[][] matrix = {{1,1,1},{1,0,1},{1,1,1}};
        int[][] ret = setZeroes(matrix);
        for(int i=0; i<ret.length; i++){
            for(int j=0; j<ret[0].length; j++){
                System.out.print(ret[i][j] + "  ");
            }
            System.out.println("");
        }
    }
    public static int[][] setZeroes(int[][] matrix) {
        Set<Integer> set_row = new HashSet<>();
        Set<Integer> set_col = new HashSet<>();
        int row = matrix.length;
        int col = matrix[0].length;
        for(int i=0; i<row; i++){
            for(int j=0; j<col; j++){
                if(matrix[i][j]==0){
                    set_row.add(i);
                    set_col.add(j);
                }
            }
        }
        for(int i=0; i<row; i++){
            for(int j=0; j<col; j++){
                if(set_row.contains(i)||set_col.contains(j)){
                    matrix[i][j] = 0;
                }
            }
        }
        return matrix;
    }
}
