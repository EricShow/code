from typing import List


class gupiao1:


    def maxValue(self, prices:List[int]) -> int:
        min_price = int(1e9)
        maxprofit = 0

        for price in prices:
            maxprofit = max(maxprofit, price - min_price)
            min_price = min(price, min_price)

        return maxprofit

if __name__ == '__main__':

    prices = [7,1,5,3,6,4]
    gupiao1 = gupiao1()
    ret = gupiao1.maxValue(prices)
    print("ret: ", ret)