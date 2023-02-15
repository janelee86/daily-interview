

## 1. 快速排序
快速排序使用分治法（Divide and conquer）策略来把一个序列（list）分为较小和较大的2个子序列，然后递归地排序两个子序列。

步骤为：

挑选基准值：从数列中挑出一个元素，称为"基准"（pivot）;
分割：重新排序数列，所有比基准值小的元素摆放在基准前面，所有比基准值大的元素摆在基准后面（与基准值相等的数可以到任何一边）。在这个分割结束之后，对基准值的排序就已经完成;
递归排序子序列：递归地将小于基准值元素的子序列和大于基准值元素的子序列排序。
递归到最底部的判断条件是数列的大小是零或一，此时该数列显然已经有序。

选取基准值有数种具体方法，此选取方法对排序的时间性能有决定性影响。

> 题目: 这是面试中最常见的问题，手写快排，面试官主要是考查候选人的算法基本工。
> 公司: 爱奇艺，某金融公司


def quick_order(arr):
    if len(arr)<2:
        return arr
    else:
        pivot = arr[0]
        less = [i for i in arr[1:] if i <pivot]
        great = [i for i in arr[1:] if i >pivot]
        return quick_order(less)+[qq] + quick_order(great)
quick_order([1,4,2])




## 2. 堆排序
> 题目: 手写堆排序
> 公司: 阿里
> 
堆排序（Heapsort）是指利用堆这种数据结构所设计的一种排序算法。堆积是一个近似完全二叉树的结构，并同时满足堆积的性质：即子结点的键值或索引总是小于（或者大于）它的父节点。堆排序可以说是一种利用堆的概念来排序的选择排序。
```
// 第一步建立最大堆， 下标从1开始 A[1..n]
def OneHeapSort(arr, l, r):
    
    if r-l<=0:
        return
    else:
        middle = l + (r-l-1)//2
        OneHeapSort(arr, l, middle)
        OneHeapSort(arr, middle+1, r)
        if arr[middle]>arr[r]:
            arr[middle],arr[r] = arr[r],arr[middle]

# 依次将最大值放到数组的后面
def heapSort(arr):
    for i in range(len(arr)-1, 0, -1):
        OneHeapSort(arr, 0, i)
    
    return arr


list1 = [1, 10, 8, 200, 50, 4]
heapSort(list1)
```

## 3. 归并排序
> 题目: 手写归并排序
归并排序（英文：Merge sort，或mergesort），是创建在归并操作上的一种有效的排序算法。该算法是采用分治法（divide and Conquer）的一个非常典范的应用。

分治法：

分割：递归地把当序列表平均分割成两半。
集合：在保持元素顺序的同时将上一步到的子顺序集合到一起（归并）。
```
def merge_sort(lst):
    if len(lst) <= 1:
        return lst
    middle = int (len(lst)/2)

    left = merge_sort(lst[ :middle])#左边
    right = merge_sort(lst[middle: ])#右边
    merged = []
    while left and right:
        merged.append(left.pop(0) if left [0] <= right[0] else right.pop(0))
    merged.extend(right if right else left)  #该方法没有返回值，但会在已存在的列表中添加新的列表内容
    return merged
data_lst = [6,202,100,301,38,8,1]
print(merge_sort(data_lst))
```

## 4. 实现多路归并排序
> 题目: 实现常用的多路归并排序(使用最大堆，或者优先队列)
> 公司: 百度，360

```

```

## 5. 插入排序
> 题目: 插入排序（升序）。
> 公司: 百度

```
def insertionSort(arr):
    #从要排序的列表第二个元素开始比较
    for i in range(1,len(arr)):
        j = i
        #从大到小比较，直到比较到第一个元素
        while j > 0:
            if arr[j] < arr[j-1]:
                arr[j-1],arr[j] = arr[j],arr[j-1]
            j -= 1        
    return arr

arr = [1,12,2, 11, 13, 5, 6,18,4,9,-5,3,11] 
print(insertionSort(arr))

```



