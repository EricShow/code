from typing import List


class leetcode401:

    def readBinaryWatch(self, turnedOn: int) -> List[str]:
        ret = []
        for i in range(12):
            for j in range(60):
                if bin(i).count('1') + bin(j).count('1') == turnedOn:
                    ch_i = str(i)
                    ch_j = '0' + str(j) if j<10 else str(j)
                    ret.append(ch_i + ':' + ch_j)
                    #ret.append(f"{i}:{j:02d}")
        return ret

if __name__ == '__main__':
    turnedOn = int(input().strip())
    leetcode401 = leetcode401()
    print(leetcode401.readBinaryWatch(turnedOn))