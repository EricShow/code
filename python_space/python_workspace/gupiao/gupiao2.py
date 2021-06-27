from typing import List


class gupiao2:


    def maxProfit(self, prices: List[int]) -> int:
        minprice = prices[0]
        maxprofit = 0
        for i in range(1,len(prices)):
            if prices[i] > minprice:
                maxprofit += prices[i] - minprice
                minprice = prices[i]
            else:
                minprice = prices[i]
        return maxprofit

if __name__ == '__main__':

    prices = [7,1,5,3,6,4]
    gupiao2 = gupiao2()
    ret = gupiao2.maxProfit(prices)
    print("ret: ", ret)