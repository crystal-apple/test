#给定一个 n × n 的二维矩阵表示一个图像。将图像顺时针旋转 90 度。
#list=[[1,2,3],[4,5,6],[7,8,9]]
'''
#第一种
def rotate(A):
    length = len(A)
    for i in range(length):
        for j in range(i+1,length):
            temp = A[i][j] 
            A[i][j]=A[j][i]
            A[j][i]=temp
    return A  
if __name__=="__main__":
    #matrix=zip([1,2,3],[4,5,6],[7,8,9])
    matrix=[[1,2,3],[4,5,6],[7,8,9]]
    num = rotate(matrix)

    for i in range(len(num)):
        num[i]=num[i][::-1]
    print(num)
'''      
#第二种使用zip()函数
def rotate(A):
    A[:]=map(list,zip(*A[::-1]))
    return A
if __name__=="__main__":
    matrix=[[1,2,3],[4,5,6],[7,8,9]]
    matrix = rotate(matrix)
    print(matrix)
    print(list(matrix))
    