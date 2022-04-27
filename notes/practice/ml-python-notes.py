# a = 3
# print(a)

# a = "abc"
# print(a)

# a = 4
# b = 5
# print(a+b)



# my_list = [10,20,30,40]
# print(my_list[0])
# print(my_list[1])
# print(my_list[-1])  # to print last eliment.

# my_list_2 = [10, "abc", 30, 40]
# print(my_list_2[-3])

# if 3 > 4:
#     print("within if loop")
    
# for i in range(10):
#         print(i)
        
# def calculateSum(a,b):
#     return a+b
# print(calculateSum(3, 4))


# def calculateSumAndDivsion(a,b):
#     return a+b, a/b

# var1, var2 = calculateSumAndDivsion(10, 2)
# print(var1)
# print(var2)



# with open("my_file_1.txt","w") as f:
#     f.write("sample content 1")
    
# with open("my_file_1.txt","a") as f:
#     f.write("\nsample content 3")
    
    
    
# import numpy as np
# sample_list = [10,20,30,40,50,60]

# sample_numpy_1d_array = np.array(sample_list)

# sample_numpy_2d_array = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])

# # reshape the array
# new_arr = sample_numpy_2d_array.reshape(3,4)
# new_arr2 = sample_numpy_2d_array.reshape(1,-1)  # 1 row n columns

# new_arr2 = sample_numpy_2d_array.reshape(-1,1)  # n rows, 1 column

# new_sample = sample_numpy_2d_array[1:3,2:4]



#### pandas
import pandas as pd
 
# sample_series = pd.Series([10,20,30,40])

# sample_series_2 = pd.Series ([10,20,30,40],['a','b','c','d'])
# # sample_series_2[2]
# sample_series_2['c']

# sample_dataframe = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
# sample_dataframe1 = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],['row1','row2','row3','row4'],['col1','col2','col3'])
 
# sample_dataframe1['col3']
# sample_dataframe1[['col1','col3']]
    
# df = pd.read_csv("mycsvfile.csv")
# df.describe()
# df.info()
# df.head()
# df.head(2)


### matplot
import matplotlib.pyplot as plt

x = [1,2,3,4,5,6,7,8,9,10]
y = [10,25,35,40,50,60,80,90,95,100]

# plt.plot(x,y)

plt.scatter(x,y)

plt.xlabel('x')
plt.ylabel('y')
plt.title('sample plot')


plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Salary Analysis')
df = pd.read_csv('mycsvfile.csv')
plt.plot(df['Age'],df['Salary'])







