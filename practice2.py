# (SECTION 2) --- more on list

list_me = ["bag", "pack", "rice", "pen"] 
print(list_me[0])
print(list_me[1])
print(list_me[2])
print(list_me[3])

# add an object 
list_me.append("shoe")
print(list_me)


# remove an object 
list_me.remove("rice")
print(list_me)

# change an object 
list_me[1] = "boys"
list_me[0] = "bag updated"
list_me[2] = "rice updated"
print(list_me)

# count all objects
print(len(list_me))
for object in list_me:
    print(object)

# Quick Exercise 
fruits = ["apple", "banana", "orange"]
fruits.append("grape")
fruits.remove("banana")
print(fruits)

numbers = ["3", "7", "10", "14"]
numbers[2] = "14"
print(numbers)
print(len(numbers))

count_toys = ["car", "ball", "car", "doll", "car"]
print(len("car"))

# REVERSE LIST
nums = [1, 2, 3, 4, 5]

reversed_nums = nums[::-1]
print(reversed_nums)

# LARGEST NUMBER
print(max(nums))

# oR

largest = nums[0]
for num in nums:
    if num > largest:
        largest = num
print(largest)        