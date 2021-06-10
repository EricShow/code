import java.util.*;

class leetcode36{
    
    public static void main(String[] args) {
        char[][] board =  {{'5','3','.','.','7','.','.','.','.'}
                          ,{'6','.','.','1','9','5','.','.','.'}
                          ,{'.','9','8','.','.','.','.','6','.'}
                          ,{'8','.','.','.','6','.','.','.','3'}
                          ,{'4','.','.','8','.','3','.','.','1'}
                          ,{'7','.','.','.','2','.','.','.','6'}
                          ,{'.','6','.','.','.','.','2','8','.'}
                          ,{'.','.','.','4','1','9','.','.','5'}
                          ,{'.','.','.','.','8','.','.','7','9'}};
        //System.out.println(board);
        System.out.println(f(board));
    }
    public static boolean f(char[][] board){
        HashMap<Integer,Integer>[] map_col = new HashMap[9];
        HashMap<Integer,Integer>[] map_row = new HashMap[9];
        HashMap<Integer,Integer>[] map_box = new HashMap[9];
        for(int i=0; i<9; i++){
            map_box[i] = new HashMap<>();
            map_col[i] = new HashMap<>();
            map_row[i] = new HashMap<>();
        }
        for(int i=0; i<9; i++){
            for(int j=0; j<9; j++){
                char num = board[i][j];
                int box_index = (i/3)*3+(j/3);
                if(num!='.'){
                    int n = (int)num;
                    map_col[i].put(n,map_col[i].getOrDefault(n, 0)+1);
                    map_row[j].put(n,map_row[j].getOrDefault(n, 0)+1);
                    map_box[box_index].put(n,map_box[box_index].getOrDefault(n, 0)+1);
                    if(map_box[box_index].get(n)>1||map_row[j].get(n)>1||map_col[i].get(n)>1){
                        return false;
                    }
                }
                
            }
        }
        return true;
    }















    public static boolean isValidSudoku(char[][] board) {
        HashMap<Integer, Integer> [] rows = new HashMap[9];
        HashMap<Integer, Integer> [] columns = new HashMap[9];
        HashMap<Integer, Integer> [] boxes = new HashMap[9];
        for (int i = 0; i < 9; i++) {
            rows[i] = new HashMap<Integer, Integer>();
            columns[i] = new HashMap<Integer, Integer>();
            boxes[i] = new HashMap<Integer, Integer>();
        }
        for(int i=0; i<9; i++){
            for(int j=0; j<9; j++){
                char num = board[i][j];
                if (num != '.') {
                    int n = (int)num;
                    int box_index = (i / 3 ) * 3 + j / 3;

                    // keep the current cell value
                    rows[i].put(n, rows[i].getOrDefault(n, 0) + 1);
                    columns[j].put(n, columns[j].getOrDefault(n, 0) + 1);
                    boxes[box_index].put(n, boxes[box_index].getOrDefault(n, 0) + 1);

                    // check if this value has been already seen before
                    if (rows[i].get(n) > 1 || columns[j].get(n) > 1 || boxes[box_index].get(n) > 1)
                        return false;
                }
            }
        }
        return true;
    }
}