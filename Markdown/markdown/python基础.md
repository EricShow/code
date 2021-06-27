https://blog.csdn.net/zhangshengdong1/article/details/95917124?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522162106867916780261984739%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=162106867916780261984739&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-95917124.first_rank_v2_pc_rank_v29&utm_term=python+%E9%94%AE%E7%9B%98%E8%BE%93%E5%85%A5





### python键盘输入

1.输入单个数字
(1)sys.stdin.readline()方法

```python
import sys
a = int(sys.stdin.readline().strip())  #不能输入222 333多个数字在一行
b = int(a)
```

(2)input()方法

```python
a = input()
b = int(a)
```

2.输入多个数字(输入时用空格隔开

```python
import sys
#这样写循环退不出来
#input: 222 333 output:['1', '2', '3']
for line in sys.stdin:
    a = line.split()
for line in sys.stdin:  #input: 1 2 3  output: in for circle:  ['1', '2', '3'] [1, 2, 3]
    a = line.split()
    ls = [int(a[i]) for i in range(len(a))]
    print("in for circle: ", a, ls)
```

3.输入多行

```python
import sys
n = int(sys.stdin.readline().strip())
ans = 0
for i in range(n):
	line = sys.stdin.readline().strip()
	values = list(map(int,line.strip()))
	for v in values:
		ans += v
print(ans)

```

最后还需要注意的是，在jupyter以及spyder等ipython编辑器中无法直接运行这类键盘输入的程序，需要转到命令行使用`python xxx.py`的命令运行。

### python类

```python
class solution:
    def twoSum(self, nums: List[int],target: int)-> List[int]:
        hashtable = dict()
        #print(type(target))
        for i, num in enumerate(nums):
            #print(type(num))
            if (target - num) in hashtable:
                return [hashtable[target - num], i]
            hashtable[int(nums[i])] = i
        return []
    
#键盘输入
a = input('列表：').split(" ")
ls = [int(a[i]) for i in range(len(a))]
target = int(input())
s = solution() #注意这个括号，不带括号叫赋值，带括号叫实例化
print(s.twoSum(nums = ls, target = target))
```





### Python字典dict方法

**1、dict.clear()**

clear() 用于清空字典中所有元素（键-值对），对一个字典执行 clear() 方法之后，该字典就会变成一个空字典。

```python
a={"person1":{"Andy":30},"person12":{"Lady":45}}
a.clear()
print(a)
#output: {}
```

**2、dict.copy()**

copy() 用于返回一个字典的浅拷贝。

```python
list1 = ['Author', 'age', 'sex']
list2 = ['Python当打之年', 99, '男']
dic1 = dict(zip(list1, list2))
 
dic2 = dic1 # 浅拷贝: 引用对象
dic3 = dic1.copy() # 浅拷贝：深拷贝父对象（一级目录），子对象（二级目录）不拷贝，还是引用
dic1['age'] = 18
 
# dic1 = {'Author': 'Python当打之年', 'age': 18, 'sex': '男'}
# dic2 = {'Author': 'Python当打之年', 'age': 18, 'sex': '男'}
# dic3 = {'Author': 'Python当打之年', 'age': 99, 'sex': '男'}
```

其中 dic2 是 dic1 的引用，所以输出结果是一致的，dic3 父对象进行了深拷贝，不会随dic1 修改而修改，子对象是浅拷贝所以随 dic1 的修改而修改，注意父子关系。

**拓展深拷贝：copy.deepcopy()**

```python
import copy
 
list1 = ['Author', 'age', 'sex']
list2 = ['Python当打之年', [18,99], '男']
dic1 = dict(zip(list1, list2))
 
dic2 = dic1
dic3 = dic1.copy()
dic4 = copy.deepcopy(dic1)
dic1['age'].remove(18)
dic1['age'] = 20
 
# dic1 = {'Author': 'Python当打之年', 'age': 20, 'sex': '男'}
# dic2 = {'Author': 'Python当打之年', 'age': 20, 'sex': '男'}
# dic3 = {'Author': 'Python当打之年', 'age': [99], 'sex': '男'}
# dic4 = {'Author': 'Python当打之年', 'age': [18, 99], 'sex': '男'}
```

dic2 是 dic1 的引用，所以输出结果是一致的；dic3 父对象进行了深拷贝，不会随dic1 修改而修改，子对象是浅拷贝所以随 dic1 的修改而修改；dic4 进行了深拷贝，递归拷贝所有数据，相当于完全在另外内存中新建原字典，所以修改dic1不会影响dic4的数据

**3、dict.fromkeys()**

fromkeys() 使用给定的多个键创建一个新字典，值默认都是 None，也可以传入一个参数作为默认的值。

```python
list1 = ['Author', 'age', 'sex']
dic1 = dict.fromkeys(list1)
dic2 = dict.fromkeys(list1, 'Python当打之年')
 
# dic1 = {'Author': None, 'age': None, 'sex': None}
# dic2 = {'Author': 'Python当打之年', 'age': 'Python当打之年', 'sex': 'Python当打之年'}
```

**4、dict.get()**

get() 用于返回指定键的值，也就是根据键来获取值，在键不存在的情况下，返回 None，也可以指定返回值。

```python
list1 = ['Author', 'age', 'sex']
list2 = ['Python当打之年', [18,99], '男']
dic1 = dict(zip(list1, list2))
 
Author = dic1.get('Author')
# Author = Python当打之年
phone = dic1.get('phone')
# phone = None
phone = dic1.get('phone','12345678')
# phone = 12345678
```

**5、dict.items()**

items() 获取字典中的所有键-值对，一般情况下可以将结果转化为列表再进行后续处理。

```python

list1 = ['Author', 'age', 'sex']
list2 = ['Python当打之年', [18,99], '男']
dic1 = dict(zip(list1, list2))
items = dic1.items()
print('items = ', items)
print(type(items))
print('items = ', list(items))
 
# items = dict_items([('Author', 'Python当打之年'), ('age', [18, 99]), ('sex', '男')])
# <class 'dict_items'>
# items = [('Author', 'Python当打之年'), ('age', [18, 99]), ('sex', '男')]
```

**6、dict.keys()**

keys() 返回一个字典所有的键

```python
list1 = ['Author', 'age', 'sex']
list2 = ['Python当打之年', [18,99], '男']
dic1 = dict(zip(list1, list2))
keys = dic1.keys()
print('keys = ', keys)
print(type(keys))
print('keys = ', list(keys))
 
# keys = dict_keys(['Author', 'age', 'sex'])
# <class 'dict_keys'>
# keys = ['Author', 'age', 'sex']
```

**7、dict.pop()**

pop() 返回指定键对应的值，并在原字典中删除这个键-值对。

```python
list1 = ['Author', 'age', 'sex']
list2 = ['Python当打之年', [18,99], '男']
dic1 = dict(zip(list1, list2))
sex = dic1.pop('sex')
print('sex = ', sex)
print('dic1 = ',dic1)
# 注释：pop必须list1中的key
# sex = 男
# dic1 = {'Author': 'Python当打之年', 'age': [18, 99]}
```

**8、dict.popitem()**

popitem() 删除字典中的最后一对键和值。

```php
list1 = ['Author', 'age', 'sex']
list2 = ['Python当打之年', [18,99], '男']
dic1 = dict(zip(list1, list2))
dic1.popitem()
print('dic1 = ',dic1) 
 
# dic1 = {'Author': 'Python当打之年', 'age': [18, 99]}
```

**9、dict.setdefault()**

setdefault() 和 get() 类似, 但如果键不存在于字典中，将会添加键并将值设为default。

```python
list1 = ['Author', 'age', 'sex']
list2 = ['Python当打之年', [18,99], '男']
dic1 = dict(zip(list1, list2))
dic1.setdefault('Author', '当打之年')
print('dic1 = ',dic1)
# dic1 = {'Author': 'Python当打之年', 'age': [18, 99], 'sex': '男'}
dic1.setdefault('name', '当打之年')
print('dic1 = ',dic1)
# dic1 = {'Author': 'Python当打之年', 'age': [18, 99], 'sex': '男', 'name': '当打之年'}
```

**10、dict.update(dict1)**

update() 字典更新，将字典dict1的键-值对更新到dict里，如果被更新的字典中己包含对应的键-值对，那么原键-值对会被覆盖，如果被更新的字典中不包含对应的键-值对，则添加该键-值对。

```python
list1 = ['Author', 'age', 'sex']
list2 = ['Python当打之年', [18,99], '男']
dic1 = dict(zip(list1, list2))
print('dic1 = ',dic1)
# dic1 = {'Author': 'Python当打之年', 'age': [18, 99], 'sex': '男'}
 
list3 = ['Author', 'phone' , 100]
list4 = ['当打之年', 12345678, 0]
dic2 = dict(zip(list3, list4))
print('dic2 = ',dic2)
# dic2 = {'Author': '当打之年', 'phone': 12345678}
# 字典1中有了Author，因此会更新Author对应的value，而字典1中没有的python,则会更新上去
dic1.update(dic2)
print('dic1 = ',dic1)
# dic1 = {'Author': '当打之年', 'age': [18, 99], 'sex': '男', 'phone': 12345678}
```

**11、dict.values()**

values() 返回一个字典所有的值。

```python
list1 = ['Author', 'age', 'sex']
list2 = ['Python当打之年', [18, 99], '男']
dic1 = dict(zip(list1, list2))
print(len(dic1))    #output: 3
values = dic1.values()
print('values = ', values)
print(type(values))
print('values = ', list(values))

# values = dict_values(['Python当打之年', [18, 99], '男'])
# <class 'dict_values'>
# values = ['Python当打之年', [18, 99], '男']
```



### Python List

 列表是Python中最基本的数据结构，列表是最常用的Python数据类型，列表的数据项不需要具有相同的类型。列表中的每个元素都分配一个数字 - 它的位置，或索引，第一个索引是0，第二个索引是1，依此类推。 
       Python有6个序列的内置类型，但最常见的是列表和元组。序列都可以进行的操作包括索引，切片，加，乘，检查成员。此外，Python已经内置确定序列的长度以及确定最大和最小的元素的方法。

一、创建一个列表 
只要把逗号分隔的不同的数据项使用方括号括起来即可。如下所示：

```python
list1 = ['physics', 'chemistry', 1997, 2000];
list2 = [1, 2, 3, 4, 5 ];
list3 = ["a", "b", "c", "d"];
```

与字符串的索引一样，列表索引从0开始。列表可以进行截取、组合等。 
**二、访问列表中的值** 
使用下标索引来访问列表中的值，同样你也可以使用方括号的形式截取字符，如下所示：

```python
list1 = ['physics', 'chemistry', 1997, 2000];
list2 = [1, 2, 3, 4, 5, 6, 7 ];
 
print "list1[0]: ", list1[0]
print "list2[1:5]: ", list2[1:5]

#output
list1[0]:  physics
list2[1:5]:  [2, 3, 4, 5]
```

**三、更新列表** 

你可以对列表的数据项进行修改或更新，你也可以使用append()方法来添加列表项，如下所示：

```python

list = ['physics', 'chemistry', 1997, 2000];
print "Value available at index 2 : "
print list[2];
list[2] = 2001;
print "New value available at index 2 : "
print list[2];
# output 
Value available at index 2 :
1997
New value available at index 2 :
2001
```

**四、删除列表元素** 

可以使用 del 语句来删除列表的的元素，如下实例：

```python
list1 = ['physics', 'chemistry', 1997, 2000];
print list1;
del list1[2];
print "After deleting value at index 2 : "
print list1;

# output
['physics', 'chemistry', 1997, 2000]
After deleting value at index 2 :
['physics', 'chemistry', 2000]
```

**五、list常用操作** 

|        **Python表达式**        |         **结果**         |     **描述**     |
| :----------------------------: | :----------------------: | :--------------: |
|       len([1, 2, 3，4])        |            4             |    求list长度    |
|  [1, 2, 3] + ['a', 'b', 'c']   | [1, 2, 3, 'a', 'b', 'c'] | “+”实际上是连接  |
|           ['a'] * 3            |      ['a','a','a']       | “*” 实际上是复制 |
|       3 in [1, 2, 3, 4]        |           True           | 检查成员是否存在 |
| for i in [1, 2, 3, 4] print(x) |         1 2 3 4          |       迭代       |
**六、Python list函数&方法**

Python内置以下操作list的函数：

|       函数        |                          说明                           |
| :---------------: | :-----------------------------------------------------: |
| cmp(list1, list2) | 比较两个列表的元素,比较方法与其他语言字符串的比较相同。 |
|    `len(list)`    |                    求列表元素个数。                     |
|    `max(list)`    |                   返回列表元素最大值                    |
|    `min(list)`    |                   返回列表元素最小值                    |
|   `list(tuple)`   |                    将元组转换为列表                     |

Python list包含以下方法:

|          方法           |                             说明                             |
| :---------------------: | :----------------------------------------------------------: |
|    list.append(obj)     |                    在列表末尾添加新的对象                    |
|     list.count(obj)     |                统计某个元素在列表中出现的次数                |
|     list.index(obj)     |            列表中找出某个值第一个匹配项的索引位置            |
| list.insert(index, obj) |                        将对象插入列表                        |
| list.pop(obj=list[-1])  | 移除列表中的一个元素（默认最后一个元素），并且返回该元素的值 |
|    list.remove(obj)     |                移除列表中某个值的第一个匹配项                |
|     list.reverse()      |                        反向列表中元素                        |
|    list.sort([func])    |                       对原列表进行排序                       |

list.insert()

```python
lst = [2,2,2,2,2,2]
lst.insert(-1,6)
print(lst)
#output 这样并不是在最后的位置增加一个元素
[2, 2, 2, 2, 2, 6, 2]

lst = [2,2,2,2,2,2]
lst.insert(30,20)# 当index >= len(list)时，从尾部插入obj
print(lst)
# output
[2, 2, 2, 2, 2, 2, 20

```



list.count()方法

```python
aList = [123, 'xyz', 'zara', 'abc', 123]
print "Count for 123 : ", aList.count(123)
print "Count for zara : ", aList.count('zara')
# output
Count for 123 : 2 
Count for zara : 1
```



list.sort()方法

```python
# 获取列表的第二个元素
def takeSecond(elem):
    return elem[1]
    
# 列表
random = [(2, 2), (3, 4), (4, 1), (1, 3)]
# 指定第二个元素排序
random.sort(key=takeSecond)
 
# 输出类别
print ('排序列表：', random)
排序列表：[(4, 1), (2, 2), (1, 3), (3, 4)]
```

|      名称       |                             说明                             |                     备注                     |
| :-------------: | :----------------------------------------------------------: | :------------------------------------------: |
|      list       |                           列表名称                           |                                              |
|  key=function   | key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。 | 可省略的参数。省略时列表排序不考虑元素的权值 |
| reverse=boolean | reverse -- 排序规则，reverse = True 降序，reverse = False 升序（默认）。 |    可省略的参数。省略时reverse默认值False    |

### Python中set()集合的使用方法

python中的set()是一个无序的不重复的元素集。

sets 支持 x in set, len(set),和 for x in set。

1.初始化方法：

```python
x = set()
x.add('str')
......
```

2.基本使用方法：

```python
x = set('class')
y = set(['c','a','m'])
print(x, y)
>>>(set(['l', 'c', 'a', 's']), set(['c', 'a', 'm']))
```

储存方式是==无序的==，==不重复的==。

3.交、并、差集

```python
#交集
print(x&y)
>>>set(['c', 'a'])
#并集
print(x | y)
>>>set(['l', 'c', 'a', 's', 'm'])
#差集
print（x - y）
>>>set(['l','s'])
```

4.去除list列表中重复的数据

在某些长列表中，需要获取列表中的元素类型时，可以使用set()方法去除重复的元素。

```python
a = [1,2,3,4,3,1]
b = set(a)
print(b)
>>>set([2,1,3,4])
```

与列表和元组不同，集合是无序的，也无法通过数字进行索引。此外，集合中的元素不能重复。

5.基本操作方式：

```python
# 添加一项
a.add('x')
# 在a中添加多项
a.update([10,37,42])
#使用remove()可以删除一项,如果不存在则引发 KeyError
a.remove('c')
#获取集合的长度(元素个数)
len(a)
#测试是否为成员项
'2' in a
#测试是否不为成员项
'2' not in a
#测试是否 s 中的每一个元素都在 t 中
s.issubset(t)
s <= t
#测试是否 t 中的每一个元素都在 s 中
s.issuperset(t)
s >= t
#返回一个新的 set 包含 s 和 t 中的每一个元素
s.union(t)
s | t
#返回一个新的 set 包含 s 和 t 中的公共元素
s.intersection(t)
s & t
#返回一个新的 set 包含 s 中有但是 t 中没有的元素
s.difference(t)
s - t
#返回一个新的 set 包含 s 和 t 中不重复的元素
s.symmetric_difference(t)
s ^ t
#返回 set “s”的一个浅复制
s.copy()
#删除并且返回 set “s”中的一个不确定的元素, 如果为空则引KeyError
s.pop()
#删除 set “s”中的所有元素
s.clear()
#如果在 set “s”中存在元素 x, 则删除
s.discard(x)
```

### Python set集合方法详解

|            方法名             |                语法格式                |                             功能                             | 实例                                                         |
| :---------------------------: | :------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------- |
|             add()             |               set1.add()               |       向 set1 集合中添加数字、字符串、元组或者布尔类型       | >>> set1 = {1,2,3}   <br>>>> set1.add((1,2)) <br/>>>> set1 {(1, 2), 1, 2, 3} |
|            clear()            |              set1.clear()              |                   清空 set1 集合中所有元素                   | >>> set1 = {1,2,3}<br/>>>> set1.clear() <br/>>>> set1 set()<br/> set()才表示空集合，{}表示的是空字典 |
|            copy()             |           set2 = set1.copy()           |                    拷贝 set1 集合给 set2                     | \>>> set1 = {1,2,3}<br/>\>>> set2 = {3,4}<br/>\>>> set3 = set1.difference(set2)<br/>\>>> set3<br/>{1, 2} |
|      difference_update()      |      set1.difference_update(set2)      |               从 set1 中删除与 set2 相同的元素               | \>>> set1 = {1,2,3}<br/>\>>> set2 = {3,4}<br/>\>>> set1.difference_update(set2)<br/>\>>> set1<br/>{1, 2} |
|           discard()           |           set1.discard(elem)           |                   删除 set1 中的 elem 元素                   | \>>> set1 = {1,2,3}<br/>\>>> set1.discard(2)<br/>\>>> set1<br/>{1, 3}<br/>\>>> set1.discard(4)<br/>{1, 3} |
|        intersection()         |     set3 = set1.intersection(set2)     |                取 set1 和 set2 的交集给 set3                 | \>>> set1 = {1,2,3}<br/>\>>> set2 = {3,4}<br/>\>>> set3 = set1.intersection(set2)<br/>\>>> set3<br/>{3} |
|     intersection_update()     |     set1.intersection_update(set2)     |             取 set1和 set2 的交集，并更新给 set1             | \>>> set1 = {1,2,3}<br/>\>>> set2 = {3,4}<br/>\>>> set1.intersection_update(set2)<br/>\>>> set1<br/>{3} |
|         isdisjoint()          |         set1.isdisjoint(set2)          | 判断 set1 和 set2 是否没有交集，有交集返回 False；没有交集返回 True | \>>> set1 = {1,2,3}<br/>\>>> set2 = {3,4}<br/>\>>> set1.isdisjoint(set2)<br/>False |
|          issubset()           |          set1.issubset(set2)           |                 判断 set1 是否是 set2 的子集                 | \>>> set1 = {1,2,3}<br/>\>>> set2 = {1,2}<br/>\>>> set1.issubset(set2)<br/>False |
|         issuperset()          |         set1.issuperset(set2)          |                 判断 set2 是否是 set1 的子集                 | \>>> set1 = {1,2,3}<br/>\>>> set2 = {1,2}<br/>\>>> set1.issuperset(set2)<br/>True |
|             pop()             |             a = set1.pop()             |                取 set1 中一个元素，并赋值给 a                | \>>> set1 = {1,2,3}<br/>\>>> a = set1.pop()<br/>\>>> set1<br/>{2,3}<br/>\>>> a<br/>1 |
|           remove()            |           set1.remove(elem)            |                   移除 set1 中的 elem 元素                   | >>> set1 = {1,2,3}<br/>>>> set1.remove(2)<br/>>>> set1<br/>{1, 3}<br/>>>> set1.remove(4)<br/>Traceback (most recent call last):<br/>  File "<pyshell#90>", line 1, in <module><br/>    set1.remove(4)<br/>KeyError: 4<br/> |
|    symmetric_difference()     | set3 = set1.symmetric_difference(set2) |          取 set1 和 set2 中互不相同的元素，给 set3           | \>>> set1 = {1,2,3}<br/>\>>> set2 = {3,4}<br/>\>>> set3 = set1.symmetric_difference(set2)<br/>\>>> set3<br/>{1, 2, 4} |
| symmetric_difference_update() | set1.symmetric_difference_update(set2) |       取 set1 和 set2 中互不相同的元素，并更新给 set1        | \>>> set1 = {1,2,3}<br/>\>>> set2 = {3,4}<br/>\>>> set1.symmetric_difference_update(set2)<br/>\>>> set1<br/>{1, 2, 4} |
|            union()            |        set3 = set1.union(set2)         |              取 set1 和 set2 的并集，赋给 set3               | \>>> set1 = {1,2,3}<br/>\>>> a = set1.pop()<br/>\>>> set1<br/>{2,3}<br/>\>>> a<br/>1 |
|           remove()            |           set1.remove(elem)            |                   移除 set1 中的 elem 元素                   | >>> set1 = {1,2,3}<br/>>>> set1.remove(2)<br/>>>> set1<br/>{1, 3}<br/>>>> set1.remove(4)<br/>Traceback (most recent call last):<br/>  File "<pyshell#90>", line 1, in <module><br/>    set1.remove(4)<br/>KeyError: 4<br/> |
|    symmetric_difference()     | set3 = set1.symmetric_difference(set2) |          取 set1 和 set2 中互不相同的元素，给 set3           | \>>> set1 = {1,2,3}<br/>\>>> set2 = {3,4}<br/>\>>> set3 = set1.symmetric_difference(set2)<br/>\>>> set3<br/>{1, 2, 4} |
| symmetric_difference_update() | set1.symmetric_difference_update(set2) |       取 set1 和 set2 中互不相同的元素，并更新给 set1        | \>>> set1 = {1,2,3}<br/>\>>> set2 = {3,4}<br/>\>>> set1.symmetric_difference_update(set2)<br/>\>>> set1<br/>{1, 2, 4} |
|            union()            |        set3 = set1.union(set2)         |              取 set1 和 set2 的并集，赋给 set3               | \>>> set1 = {1,2,3}<br/>\>>> set2 = {3,4}<br/>\>>> set3=set1.union(set2)<br/>\>>> set3<br/>{1, 2, 3, 4} |
|           update()            |           set1.update(elem)            |                添加列表或集合中的元素到 set1                 | \>>> set1 = {1,2,3}<br/>\>>> set1.update([3,4])<br/>\>>> set1<br/>{1,2,3,4} |

### Python基础：标准库和常用的第三方库

**标准库**

|   名称   |                             作用                             |                                             |
| :------: | :----------------------------------------------------------: | ------------------------------------------- |
| datetime |         为日期和时间处理同时提供了简单和复杂的方法。         |                                             |
|   zlib   | 直接支持通用的数据打包和压缩格式：zlib，gzip，bz2，zipfile，以及 tarfile。 |                                             |
|  random  |                   提供了生成随机数的工具。                   |                                             |
|   math   |            为浮点运算提供了对底层C函数库的访问。             |                                             |
|   sys    | 工具脚本经常调用命令行参数。这些命令行参数以链表形式存储于 sys 模块的 argv 变量。 |                                             |
|   glob   |      提供了一个函数用于从目录通配符搜索中生成文件列表。      | https://docs.python.org/3/library/glob.html |
|    os    |              提供了不少与操作系统相关联的函数。              |                                             |



第三方库

| 名称          | 作用                                                         | 使用参考                                                |
| ------------- | ------------------------------------------------------------ | ------------------------------------------------------- |
|               |                                                              |                                                         |
|               |                                                              |                                                         |
| Scrapy        | 爬虫工具常用的库。                                           | https://blog.csdn.net/alice_tl/article/details/81433214 |
| Requests      | http库。                                                     |                                                         |
| Pillow        | 是PIL（Python图形库）的一个分支。适用于在图形领域工作的人。  | https://blog.csdn.net/alice_tl/article/details/80866728 |
| matplotlib    | 绘制数据图的库。对于数据科学家或分析师非常有用。             |                                                         |
| OpenCV        | 图片识别常用的库，通常在练习人脸识别时会用到                 | https://blog.csdn.net/alice_tl/article/details/89291235 |
| pytesseract   | 图片文字识别，即OCR识别                                      | https://blog.csdn.net/alice_tl/article/details/89299405 |
| wxPython      | Python的一个GUI（图形用户界面）工具。                        |                                                         |
| Twisted       | 对于网络应用开发者最重要的工具。                             |                                                         |
| SymPy         | SymPy可以做代数评测、差异化、扩展、复数等等。                |                                                         |
| SQLAlchemy    | 数据库的库。                                                 |                                                         |
| SciPy         | Python的算法和数学工具库。                                   |                                                         |
| Scapy         | 数据包探测和分析库。                                         |                                                         |
| pywin32       | 提供和windows交互的方法和类的Python库。                      |                                                         |
| pyQT          | Python的GUI工具。给Python脚本开发用户界面时次于wxPython的选择。 |                                                         |
| pyGtk         | 也是Python GUI库。                                           |                                                         |
| Pyglet        | 3D动画和游戏开发引擎。                                       |                                                         |
| Pygame        | 开发2D游戏的时候使用会有很好的效果。                         |                                                         |
| NumPy         | 为Python提供了很多高级的数学方法。                           |                                                         |
| nose          | Python的测试框架。                                           |                                                         |
| IPython       | Python的提示信息。包括完成信息、历史信息、shell功能，以及其他很多很多方面。 |                                                         |
| BeautifulSoup | xml和html的解析库，对于新手非常有用。                        |                                                         |
|               |                                                              |                                                         |
|               |                                                              |                                                         |
|               |                                                              |                                                         |
|               |                                                              |                                                         |



### Python字符串基本操作

| 名称                                      | 作用                                                         | 实例                                                         |
| ----------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| strip()                                   | 从字符串的开头和结尾处丢弃掉指定字符，<br/>默认是空格符，也可以自定其他字符 | \>>>'hello   '.strip() <br/>'hello' <br/>>>>'-----hellop======'.strip('-=') <br/>'hello' |
| lstrip()                                  | 删除开头的指定字符，默认为空格，从左边删除指定字符           | >>>'  python  '.lstrip() <br/>'python  ' <br/>>>>'    __+===python'.lstrip(' _=+') <br/>'python' |
| rstrip()                                  | 删除末尾的指定字符，从右边删除指定字符                       |                                                              |
| ljust()                                   | 左对齐，接收至多两个参数，第一个参数表示长度，第二个为填充字符，长度不够，默认以空格符填充，也可以指定其他字符 | \>>>'hello'.ljust(10) <br/>'hello     ' <br/>共5个空格10个字符<br/>>>>'hello'.ljust(10, '*') <br/>'hello\*\*\*\*\*' |
| rjust()                                   | 右对齐                                                       |                                                              |
| center()                                  |                                                              | \>>>'hello'.center(10) <br/>'  hello   ' <br/>>>>'hello'.center(10, '=') '==hello===' |
| join()                                    | 语法：  'sep'.join(seq)<br/>sep：分隔符。可以为空 <br/>seq：要连接的元素序列、字符串、元组、字典 <br/>上面的语法即：以sep作为分隔符，将seq所有的元素合并成一个新的字符串 | \>>>' '.join(['name', 'age', 'address']) <br/>'name age address'<br/>>>> seq1 = ['hello','good','boy','doiido']    <br/>>>> print ' '.join(seq1)    <br/>hello good boy doiido    <br/>>>> print ':'.join(seq1)    hello:good:boy:doiido |
| index()                                   | index() 方法检测字符串中是否包含子字符串 ，如果指定 begin 和 end 范围，则检查是否包含在指定范围内，如果存在就返回第一个配置到位置, 该方法与 find()方法一样，只不过如果str不在 string中会报一个ValueError异常，而find()会返回-1表示查找失败。 | >>>'abcabc'.index('a')<br/>0<br/>>>>'abcabc'.index('a', 2)<br/>3<br/>>>>'abcabc'.index('av')<br/>ValueError: substring not found<br/> |
| len(string)                               | 计算字符串的长度                                             | word = 'love' <br/>words = word*3 <br/>print(len(words)) <br/>*# echo* <br/>12 |
| string[left,right]                        | 字符串的分片与索引                                           | print(name[5:])			<br/># 右侧不写，代表到最后 <br/>'me is Mike' <br/>print(name[:5])<br/># 注意是左闭右开，所以第5个是取不到的 <br/>'My na' |
| string.find(sub_string)                   | 查找子字符串出现的位置                                       | sub_num = '123' <br/>phone_num = '138-1034-5123'<br/>print(phone_num.find(sub_num))<br/>10*<br/>\# 返回第一次出现123这个字符串的位置，10是1的位置* |
| string.replace(string_a,string_b)         | 替换部分字符串                                               | phone_num = '138-1034-5123' *<br/>\# 返回第一次出现123这个字符串的位置，10是1的位置*<br/>phone_num_new = phone_num.replace(phone_num[9:13],'*'*4) <br/>print(phone_num_new) <br/>*# echo* 138-1034-**** |
| str(int_a)                                | 强制类型转换                                                 |                                                              |
| {} and {}’.format(string_a,string_b)      | 字符串格式化符                                               | a = 'I' <br/>b = 'you' <br/>print('{} love {} forever.'.format(a,b)) <br/>*# echo* <br/>I love you forever. |
| strings.count(string)                     | 统计字符string出现的次数                                     | lyric = 'The night begin to shine, the night begin to shine'<br/>words = lyric.split()<br/>word = 'night'<br/>print(words.count(word))<br/><br/># echo<br/>2							<br/># 'night'出现了两次<br/> |
| strings.strip(string.punctuation).lower() | 忽略字符，转换小写                                           | 1`string.punctuation`包含以下的特殊字符：<br/>2 \!"*#$%&'()\*+,-./:;<=>?@[\]^_`{|}~*      3 使用`strip(ch)`可以忽略`ch`这些字符 <br/> 4 string.lower()`可以将字符串全部转换成小写。<br/>import string<br/>lyric = 'The night begin to shine, the night begin to shine'<br/># 首先把字符串按照空格分开，切割成包含一个个string的list<br/># 接着使用列表解析式，对每个元素进行1）忽略特殊字符2）转换成小写单词<br/>words = [word.strip(string.punctuation).lower() for word in lyric.split()]<br/>print(words)<br/><br/># echo<br/>['the', 'night', 'begin', 'to', 'shine', 'the', 'night', 'begin', 'to', 'shine']<br/> |



### Python Math常用方法总结

|          函数          |                       功能                        | 实例                                                         |
| :--------------------: | :-----------------------------------------------: | ------------------------------------------------------------ |
|         math.e         |                     自然常数e                     | >>> math.e <br/>2.718281828459045                            |
|        math.pi         |                     圆周率pi                      | >>> math.pi <br/>3.141592653589793                           |
|    math.degrees(x)     |                     弧度转度                      | >>> math.degrees(math.pi) <br/>180.0                         |
|    math.radians(x)     |                     度转弧度                      | >>> math.radians(45) <br/>0.7853981633974483                 |
|      math.exp(x)       |                   返回e的x次方                    | >>> math.exp(2) <br/>7.38905609893065                        |
|     math.expm1(x)      |                  返回e的x次方减1                  | >>> math.expm1(2) <br/>6.38905609893065                      |
|  math.log(x[, base])   |       返回x的以base为底的对数，base默认为e        | >>> math.log(math.e) <br/>1.0 <br/>>>> math.log(2, 10) <br/>0.30102999566398114 |
|     math.log10(x)      |               返回x的以10为底的对数               | >>> math.log10(2) <br/>0.30102999566398114                   |
|     math.log1p(x)      |           返回1+x的自然对数（以e为底）            | >>> math.log1p(math.e-1) <br/>1.0                            |
|     math.pow(x, y)     |                   返回x的y次方                    | >>> math.pow(5,3) <br/>125.0                                 |
|      math.sqrt(x)      |                   返回x的平方根                   | >>> math.sqrt(3) <br/>1.7320508075688772                     |
|      math.ceil(x)      |                 返回不小于x的整数                 | >>> math.ceil(5.2) <br/>6.0                                  |
|     math.floor(x)      |                 返回不大于x的整数                 | >>> math.floor(5.8) <br/>5.0                                 |
|     math.trunc(x)      |                  返回x的整数部分                  | >>> math.trunc(5.8) <br/>5                                   |
|      math.modf(x)      |                 返回x的小数和整数                 | >>> math.modf(5.2) <br/>(0.20000000000000018, 5.0)           |
|      math.fabs(x)      |                   返回x的绝对值                   | >>> math.fabs(-5) <br/>5.0                                   |
|    math.fmod(x, y)     |                  返回x%y（取余）                  | >>> math.fmod(5,2) <br/>1.0                                  |
| math.fsum([x, y, ...]) |                 返回无损精度的和                  | >>> 0.1+0.2+0.3 <br/>0.6000000000000001 <br/>>>> math.fsum([0.1, 0.2, 0.3]) <br/>0.6 |
|   math.factorial(x)    |                    返回x的阶乘                    | >>> math.factorial(5) <br/>120                               |
|     math.isinf(x)      |      若x为无穷大，返回True；否则，返回False       | >>> math.isinf(1.0e+308) <br/>False <br/>>>> math.isinf(1.0e+309) <br/>True |
|     math.isnan(x)      |      若x不是数字，返回True；否则，返回False       | >>> math.isnan(1.2e3) <br/>False                             |
|    math.hypot(x, y)    |            返回以x和y为直角边的斜边长             | >>> math.hypot(3,4) <br/>5.0                                 |
|  math.copysign(x, y)   | 若y<0，返回-1乘以x的绝对值；  否则，返回x的绝对值 | >>> math.copysign(5.2, -1) <br/>-5.2                         |
|     math.frexp(x)      |            返回m和i，满足m乘以2的i次方            | >>> math.frexp(3) <br/>(0.75, 2)                             |
|    math.ldexp(m, i)    |                 返回m乘以2的i次方                 | >>> math.ldexp(0.75, 2) <br/>3.0                             |
|      math.sin(x)       |             返回x（弧度）的三角正弦值             | >>> math.sin(math.radians(30)) <br/>0.49999999999999994      |
|      math.asin(x)      |                返回x的反三角正弦值                | >>> math.asin(0.5) <br/>0.5235987755982989                   |
|      math.cos(x)       |             返回x（弧度）的三角余弦值             | >>> math.cos(math.radians(45)) <br/>0.7071067811865476       |
|      math.acos(x)      |                返回x的反三角余弦值                | >>> math.acos(math.sqrt(2)/2) <br/>0.7853981633974483        |
|      math.tan(x)       |             返回x（弧度）的三角正切值             | >>> math.tan(math.radians(60)) <br/>1.7320508075688767       |
|      math.atan(x)      |                返回x的反三角正切值                | >>> math.atan(1.7320508075688767) <br/>1.0471975511965976    |
|    math.atan2(x, y)    |               返回x/y的反三角正切值               | >>> math.atan2(2,1) <br/>1.1071487177940904                  |
|      math.sinh(x)      |                返回x的双曲正弦函数                |                                                              |
|     math.asinh(x)      |               返回x的反双曲正弦函数               |                                                              |
|      math.cosh(x)      |                返回x的双曲余弦函数                |                                                              |
|     math.acosh(x)      |               返回x的反双曲余弦函数               |                                                              |
|      math.tanh(x)      |                返回x的双曲正切函数                |                                                              |
|     math.atanh(x)      |               返回x的反双曲正切函数               |                                                              |
|      math.erf(x)       |                  返回x的误差函数                  |                                                              |
|      math.erfc(x)      |                 返回x的余误差函数                 |                                                              |
|     math.gamma(x)      |                  返回x的伽玛函数                  |                                                              |
|     math.lgamma(x)     |         返回x的绝对值的自然对数的伽玛函数         |                                                              |



### Python OS模块的操作介绍

| 名称                           | 作用                                                         | 实例                                                         |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| os.name                        | 显示当前使用的平台                                           | **In** [1]: import os <br/>**In** [2]: os.name <br/>**Out**[2]: 'nt' |
| os.getcwd()                    | 显示当前python脚本工作路径                                   | **In** [3]: os.getcwd() <br/>**Out**[3]: 'C:\\Users\\BruceWong\\Documents' #使用Ipython |
| os.listdir(‘dirname’)          | 返回指定目录下的所有文件和目录名                             | **In** [4]: os.listdir() #如果没有指定路径，则是当前python脚本路径 <br/>**Out**[4]:<br/>['desktop.ini',<br/> 'dump.raw.txt',<br/> '自定义 Office 模板']<br/>**In** [6]: os.listdir('C:/Users/BruceWong/Documents/Notes') <br/>**Out**[6] |
| os.remove(‘filename’)          | 用于删除指定路径的文件。如果指定的路径是一个目录，将抛出OSError。强调必须是**文件**，而不是文件夹或者目录 |                                                              |
| os.makedirs(‘dirname/dirname’) | 可生成多层递规目录                                           | os.makedirs('jone/path')                                     |
| os.rmdir(‘dirname’)            | 删除单级目录                                                 | os.rmdir('jone/path')                                        |
| os.rename(“oldname”,”newname”) | 重命名文件                                                   | os.rename('jone/path','jone/pathth')                         |
| os.system()                    | 运行shell命令,注意：这里是打开一个新的shell，运行命令，当命令结束后，关闭shell。用于打开文件或者程序，不能打开文件夹 | #用于打开文件或者程序<br/>In [36]: os.system('dump.raw.txt')<br/>Out[36]: 0<br/>#注意读入程序时，需要将程序路径用引号再包括起来<br/>In [39]: os.system(r'"C:\Program Files (x86)\Tencent\WeChat\WeChat.exe"')<br/>Out[39]: 0<br/> |
| os.sep                         | 显示当前平台下路径分隔符                                     | **In** [41]: os.sep <br/>**Out**[41]: '\\'                   |
| os.linesep                     | 给出当前平台使用的行终止符                                   | **In** [42]: os.linesep <br/>**Out**[42]: '\r\n'             |
| os.environ                     | 获取系统环境变量                                             | In [43]: os.environ                                          |
| os.path.abspath(path)          | 对文件夹操作，显示当前绝对路径                               | **In** [44]: os.path.abspath('jone') #'\\'为当前的分隔符 **Out**[44]: 'C:\\Users\\BruceWong\\Documents\\jone' |
| os.path.dirname(path)          | 返回该路径的父目录                                           | In [45]: os.path.dirname(os.path.abspath('jone')) Out[45]: 'C:\\Users\\BruceWong\\Documents' |
| os.path.basename(path)         | 返回该路径的最后一个目录或者文件,如果path以／或\结尾，那么就会返回空值。 | In [46]: os.path.basename(os.path.dirname(os.path.abspath('jone'))) Out[46]: 'Documents' |
| os.path.isfile(path)           | 如果path是一个文件（file），则返回True，如果是文件夹或者目录，则返回False | **In** [49]: os.path.isfile('jone/path') <br/>**Out**[49]: **False** **<br/>In** [50]: os.path.isfile('dump.raw.txt') <br/>**Out**[50]: **True** |
| os.path.isdir(path)            | 如果path是一个目录（对应dir），则返回True                    | **In** [51]: os.path.isdir('jone/path') <br/>**Out**[51]: **True** |
| os.stat(path)                  | 获取文件或者目录信息                                         | In [53]: os.stat('dump.raw.txt')                             |
| os.path.split(path)            | 将path分割成路径名和文件名。（事实上，如果你完全使用目录，它也会将最后一个目录作为文件名而分离，同时它不会判断文件或目录是否存在） | **In** [55]: os.path.split('jone') <br/>**Out**[55]: ('', 'jone') <br/>**In** [56]: os.path.split('dump.raw.txt') <br/>**Out**[56]: ('', 'dump.raw.txt') |
| os.path.join(path,name)        | 连接目录与文件名或目录 结果为path/name，功能仅仅是连接的作用，而不能生成。将多个路径组合后返回，第一个绝对路径之前的参数将被忽略 | **In [57]**: os.path.join('jone','dump.raw.txt')<br/>**Out[57]**: 'jone\\dump.raw.txt'<br/>#上述生成的路径，如果不写入信息，则最终不会有该文件夹<br/>#不过生成的路径可以结合os.mkdir()函数生成进一步的路径。<br/>**In [69]**: path = os.path.join('jone\\path','dump.raw.txt')<br/><br/>**In [70]**: path<br/>**Out[70]**: 'jone\\path\\dump.raw.txt'<br/>**In [71]**: os.mkdir(path)<br/> |

### Python Random模块

| 名称                        | 作用                                                         | 实例                                                         |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| random.seed()               | 设置随机种子 ，用于同步不同运行环境的随机数。                | random.seed(a=None, version=2)                               |
| random.getstate()           | 获得当前状态，用于恢复状态                                   | random.getstate()                                            |
| random.random()             | 随机生成[0.1)的浮点数                                        | import random <br/>print(random.random())<br/>\#运行结果：0.4041810247152263 |
| random.uniform(a, b)        | 原型为：random.uniform(a, b)，用于生成一个指定范围内的随机符点数，两个参数其中一个是上限，一个是下限。如果a > b，则生成的随机数n: a <= n <= b。如果 a | import random <br/>print(random.uniform(10,20)) <br/>print(random.uniform(20,10)) <br/>#运行结果为19.319774059417643 #         11.25780294472681 |
| random.randint(a, b)        | 用于生成一个指定范围内的整数。其中参数a是下限，参数b是上限，生成的随机数n: **a <= n <= b**(闭区间) | print(random.randint(12, 20)) #生成的随机数n: 12 <= n <= 20  <br/>print(random.randint(20, 20))  #结果永远是20  <br/>#print(random.randint(20, 10)) #该语句是错误的。下限必须小于上限。  <br/> |
| random.choice(sequence)     | random.choice(sequence)。参数sequence表示一个**有序类型**。这里要说明 一下：sequence在python不是一种特定的类型，而是泛指一系列的类型。list, tuple, 字符串都属于sequence(字典和集合都是无序的) | import random<br/>print(random.choice("Python没那么简单") )<br/>print(random.choice(['Jason', 'is', 'so', 'handsome']))<br/>print(random.choice(('Tuple', 'List', 'Dict')))  <br/> |
| random.sample(sequence, k)  | random.sample(sequence, k)，从指定序列中随机获取指定长度的片断。(sample函数**不会修改原有序列**) | list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  <br/>slice = random.sample(list, 5)  #从list中随机获取5个元素，作为一个片断返回  <br/>print(slice)<br/>print(list) #原有序列并没有改变。 <br/>#运行结果[2, 10, 8, 7, 9]<br/>#       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]<br/> |
| random.shuffle(x[, random]) | random.shuffle(x[, random])，用于将一个列表中的元素打乱。    | import random <br/>L=[1,2,3,4,5,6,7] random.shuffle(L) <br/>print(L) <br/>#运行结果[6, 7, 2, 5, 3, 4, 1] |

```python

# -*- coding:utf-8 -*-
import random
arr = ['A','B','C','D','E','F']
#生成（0.0, 1.0）的随机数
print random.random() 
#0.133648715391
 
# 生成随机浮点数 0<N<100
print random.uniform(0,100) 
#10.535881824
 
#生成随机整数 0<N<100
print random.randint(0,100)  
 
#随机生成一个0-100内3的倍数
print random.randrange(0,100,3)
 
#29
#随机选择一个元素
print random.choice('1234567890') 
#6
print random.choice(arr) 
#B
 
#随机选择指定长度不重复元素
print random.sample('1234567890',4) 
#['3', '8', '1', '9']
print random.sample(['A','B','C','D','E','F'],4) 
#['C', 'B', 'A', 'D']
        
#打乱列表       
random.shuffle(arr)
print arr 
#['E', 'B', 'D', 'A', 'C', 'F']
```

### Python Sys模块介绍

| 名称               | 说明                                                         | 实例                                                         |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| sys.argv           | 获取当前正在执行的命令行参数的参数列表（list                 | **Input**<br/>print(sys.argv[0]) <br/>print("---------------------------------------------") <br/>for i in sys.argv:     <br/>      print(i)<br/>**Output**<br/>C:/Users/dell/Desktop/OSVOS_learning/my_code/test1.py<br/>---------------------------------------------<br/>C:/Users/dell/Desktop/OSVOS_learning/my_code/test1.py<br/> <br/>Process finished with exit code 0<br/> |
| sys.modules.keys() | 返回所有已经导入的模块列表                                   | \>>>import os <br/>\>>>import sys <br/>\>>>import numpy <br/>\>>>sys.modules.keys() <br/>dict_keys(['builtins', 'sys', '_frozen_importlib']) |
| sys.platform       | 获取当前执行环境的平台                                       | \>>> import sys <br/>\>>> sys.platform<br/>'linux2'          |
| sys.path           | path是一个目录列表，供Python从中查找第三方扩展模块           | >>>sys.path<br/>                                             |
| sys.exit(n)        | **调用sys,exit(n)可以中途退出程序**，sys.exit(0)表示正常退出，n不为0时，会引发SystemExit异常，从而在主程序中可以捕获该异常。 | import sys <br/>print("running ...") <br/>try:     <br/>          sys.exit(1) <br/>except SystemExit:     <br/>          print("SystemExit exit 1")   <br/>print("exited")<br/>**Output**：<br/>running ... <br/>SystemExit exit 1 <br/>exited<br/> |
| sys.version        | 获取python解释程序的版本信息                                 | \>>>sys.version <br/>'3.6.2 (v3.6.2:5fd33b5, Jul  8 2017, 04:57:36) [MSC v.1900 64 bit (AMD64)]' |
| sys.stdin,         | 标准输入：一般为键盘输入，stdin对象为解释器提供输入字符流，一般使用raw_input()和input()函数 | print("Please input you name:") <br/>name = sys.stdin.readline() <br/>print(name) |
| sys.stdout         | 标准输出：一般为屏幕。stdout对象接收到print语句产生的输出    | sys.stdout.write("123456\n") <br/>sys.stdout.flush() <br/>**Output**: 123456 |
|                    | 错误输出：一般是错误信息，stderr对象接收出错的信息。         | >>>raise Exception("raise...") <br/>Traceback (most recent call last):<br/>   File "<input>", line 1, in <module> <br/>Exception: raise... |
| sys.stdout         |                                                              |                                                              |
| print              |                                                              | 1 sys.stdout.write('hello'+'\n')  <br/>2 print 'hello'       |
| raw_input          | 当我们用 raw_input('Input promption: ') 时，事实上是先把提示信息输出，然后捕获输入 |                                                              |

### Python Glob模块

| 名称       | 说明                                                         | 实例                                                         |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| glob.glob  | 返回所有匹配的文件路径列表。它只有一个参数pathname，定义了文件路径匹配规则，这里可以是绝对路径，也可以是相对路径。 | **glob(pathname, recursive=False)** <br/>第一个参数pathname为需要匹配的字符串。（**该参数应尽量加上r前缀，以免发生不必要的错误**） <br/>第二个参数代表递归调用，与特殊通配符“**”一同使用，默认为False。 |
| glob.iglob | 获取一个可编历对象，使用它可以逐个获取匹配的文件路径名。与glob.glob()的区别是：glob.glob同时获取所有的匹配路径，而glob.iglob一次只获取一个匹配路径。这有点类似于.NET中操作数据库用到的DataSet与DataReader。 | 参数与glob()一致。 <br/>返回一个迭代器，该迭代器不会同时保存所有匹配到的路径，遍历该迭代器的结果与使用相同参数调用glob()的返回结果一致。 |

```python
input_dir = 'D:\py_code\dataset\Rending\\test_imgs_rendering\*.jpg'
for file in glob.glob(input_dir):
    file_name = file.split('\\')[-1].strip('.jpg')
    print (file_name)
#output：
test9_input
test9_pred_gR
test9_pred_R
test9_pred_T
for file in glob.glob(input_dir):
    file_name = file.split('\\')[-1]#删除最后一个‘\’
    print (file_name)
#output:
test9_input.jpg
test9_pred_gR.jpg
test9_pred_R.jpg
test9_pred_T.jpg
```

```python
import glob
 
listglob = []
listglob = glob.glob(r"/home/xxx/picture/*.png")
listglob.sort()
print listglob
 
print '--------------------'
listglob = glob.glob(r"/home/xxx/picture/0?.png")
listglob.sort()
print listglob
 
print '--------------------'
listglob = glob.glob(r"/home/xxx/picture/0[0,1,2].png")
listglob.sort()
print listglob
 
print '--------------------'
listglob = glob.glob(r"/home/xxx/picture/0[0-3].png")
listglob.sort()
print listglob
 
print '--------------------'
listglob = glob.iglob(r"/home/xxx/picture/0[a-z].png")
print listglob
for item in listglob:
    print item

```

### Sklearn 基础

sklearn 0.24.1官网：

https://scikit-learn.org/stable/modules/classes.html?highlight=datasets#module-sklearn.datasets



**1、random.choice()**

random.choice(container_type_object)

随机”模块的内置函数，用于从容器中返回随机元素，例如列表，字符串，元组等对象。



**2、numpy.random.uniform(low,high,size)**

函数原型： numpy.random.uniform(low,high,size)

功能：从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.

```python
np.random.uniform(-1,1,(3,))
输出：
(0.80791605 0.31561008 0.46513354)
```

**np.linalg.norm**

```
tmp_distance = np.linalg.norm(points - c_t[None, :], axis=1)
```

**None**

c_t[None,:]增加维度的操作

```python
import  numpy as np
a=np.array([[11,12,13,14],[21,22,23,24],[31,32,33,34],[41,42,43,44]])
print('0维为None:')
print(a[None,0:4])
print('1维为None:')
print(a[0:4,None])
```

```python
0维为None:
[[[11 12 13 14]
  [21 22 23 24]
  [31 32 33 34]
  [41 42 43 44]]]
1维为None:
[[[11 12 13 14]]

 [[21 22 23 24]]

 [[31 32 33 34]]

 [[41 42 43 44]]]
```

**np.argmin**





python  format