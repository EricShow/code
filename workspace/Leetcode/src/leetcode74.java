/*
搜索二位矩阵  二维矩阵的二分查找
* */
public class leetcode74 {
    public static void main(String[] args) {
        //int[][] matrix = {{1,3,5,7},{10,11,16,20},{23,30,34,60}};
        int[][] matrix = {{1,1},{2,2}};
        int target = 1;
        System.out.println(searchMatrix_official(matrix,target));
    }
    public boolean searchMatrix(int[][] matrix, int target) {
        int rowIndex = binarySearchFirstColumn(matrix, target);
        if (rowIndex < 0) {
            return false;
        }
        return binarySearchRow(matrix[rowIndex], target);
    }

    public int binarySearchFirstColumn(int[][] matrix, int target) {
        int low = -1, high = matrix.length - 1;
        while (low < high) {
            int mid = (high - low + 1) / 2 + low;
            if (matrix[mid][0] <= target) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        return low;
    }

    public boolean binarySearchRow(int[] row, int target) {
        int low = 0, high = row.length - 1;
        while (low <= high) {
            int mid = (high - low) / 2 + low;
            if (row[mid] == target) {
                return true;
            } else if (row[mid] > target) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return false;
    }



    //我的方法   通过率
    //[[1,1],[2,2]]   2   失败
    //125/133
    public static boolean searchMatrix_official(int[][] matrix, int target) {
        if(matrix.length==1){
            int left = 0;
            int right = matrix[0].length-1;
            while(left<=right){
                int mid = left + (right-left)/2;
                if(matrix[0][mid]>target){
                    right = mid-1;
                }else if(matrix[0][mid]<target) {
                    left = mid + 1;
                }else{
                    return true;
                }
            }
        }
        if(matrix[0].length==1){
            int left = 0;
            int right = matrix.length-1;
            while(left<=right){
                int mid = left + (right-left)/2;
                if(matrix[mid][0]>target){
                    right = mid-1;
                }else if(matrix[mid][0]<target) {
                    left = mid + 1;
                }else{
                    return true;
                }
            }
        }
        for(int i=1; i<matrix.length; i++){
            if(matrix[i][0]>target&&matrix[i-1][0]<=target){
                int left = 0;
                int right = matrix[0].length-1;
                while(left<=right){
                    int mid = left + (right-left)/2;
                    if(matrix[i-1][mid]>target){
                        right = mid-1;
                    }else if(matrix[i-1][mid]<target) {
                        left = mid + 1;
                    }else{
                        return true;
                    }
                }
            }
        }
        return false;
    }
}
