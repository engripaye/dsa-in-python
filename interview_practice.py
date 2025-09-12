# REVERSE A STRING 
def reverse_string(s):
    return s[::-1]

print(reverse_string("hello"))

def reverse_string2(m):
    return m[::-1]

print(reverse_string2("davido"))


# CHECK FOR PALINDROME -- A palindrome checks if a string read thesame backward and forward
def check_palindrome(s):
    return s == s[::-1]

print(check_palindrome("madam"))
print(check_palindrome("davido"))


# CHECK FOR MISSING NUMBER
def missing_number(nums):
    n = len(nums) + 1
    expected_sum = n * (n + 1) // 2
    return expected_sum - sum(nums)

print(missing_number([1, 2, 4, 5]))


# CHECK DUPLICATES 
def find_duplicates(nums):
    seen = set()
    duplicates = []
    for num in nums:
        if num in seen:
            duplicates.append(num)
        else:
            seen.add(num)
    return duplicates

print(find_duplicates([1, 2, 3, 4, 1, 4]))

# TWO SUM PROBLEMS
def two_sum_pro(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        diff = target - num
        if diff in seen:
            return [seen[diff], i]
        seen[num] = i
    return []

print(two_sum_pro([1, 2, 3, 4], 3))     


# MERGE TWO SORTED LIST
def merge_sorted_list(li, l2):
    result = []
    i = j = 0
    while i < len(li) and j < len(l2):
        if li[i] < l2[j]:
            result.append(li[i])
            i += 1
        else:
            result.append(l2[j])
            j += 1
    result.extend(li[i:])
    result.extend(l2[j:]) 
    return result

print(merge_sorted_list([1, 3, 5], [2, 4, 6]))     


# BALANCE PARENTHESIS - check if brackets are valid
def is_valid(s):
    stack = []
    mapping = {")": "(", "]": "[", "}": "{"}
    for char in s:
        if char in mapping:
            top = stack.pop() if stack else "#"
            if mapping[char] != top:
                return False
        else:
            stack.append(char) 
    return not stack

print(is_valid("()[]{}"))
print(is_valid("(]"))


# MAXIMUM SUBARRAY - find the largest sum of a subarray
def max_subarray(nums):
    max_sum = curr_sum = nums[0]
    for num in nums[1:]:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)
    return max_sum

print(max_subarray([-2,1,-3,4,-1,2,1,-5,4]))        
