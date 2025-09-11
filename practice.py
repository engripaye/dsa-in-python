# LIST
toys = ["car", "doll", "ball"]
print(toys[1])

# DICTIONARY
toy_box = {"red": "car", "blue":"doll"}
print(toy_box["red"])

# SET
unique_toy = {"car", "doll", "ball", "train"}
print(unique_toy)

# TUPLE
fixed_toys = ("car", "doll", "ball")
print(fixed_toys[1])

# ALGORITHM - are recipes for solving a problem e.g you want to 
# find a favorite toy in a list 
# Linear Search - check each item one by one
# Binary Search - cut the list in half each time  ---- (requires sorted list)

# Linear Search 
toys = ["car", "doll", "bike", "train"]
def find_toys(toys, target):
    for toy in toys:
        if toy == target:
            return "Found it!"
    return "not here!"
print(find_toys(toys, "car"))


# Linear search 2
toys2 = ["aeroplane", "jet", "helo", "cop"]
def search_object(toys2, target):
    for object in toys2:
        if object == target:
            return "Found it!"
    return "Not available!"
print(search_object(toys2, "cop"))       


# Linear Search 3
data = ["players", "coach", "board-members", "asst-coach"]
def search_data(data, target):
    for object2 in data:
        if object2 == target:
            return "found it!"
    return "not available!"
print(search_data(data, "asst-coach"))


# Linear Search 4
data2 = ["biro", "book", "ruler", "math-set"]
def search_schl_tool(tools, target):
    for tools in data2:
        if tools == target:
            return "Found it!"
    return "not available!"
print(search_schl_tool(data2, "book"))


# Binary Search 1
toyB = ["doll", "plane", "bus", "van"]

def binary_search(toyB, target):
    left, right = 0, len(toyB) - 1
    while left <= right:
        mid = (left + right) // 2
        if toyB[mid] == target:
            return "Found it!"
        elif toyB[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return "Not here"

print(binary_search(toyB, "plane"))      

# Binary search 2
music = ["babe", "fly", "bounce", "only God"]
def binary_search2(music, target):
    left, right = 0 , len(music) - 1
    while left <= right:
        mid = (left + right) // 2
        if music[mid] == target:
            return "Found it!"
        elif music[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return "Not found"                     
print(binary_search2(music, "fly"))        

# Binary search 3
artiste = ["wizkid", "davido", "Bassey", "Chioma", "sean tizzle"]
def binary_search3(artiste, target):
    left, right = 0, len(artiste)
    while left <= right:
        mid = (left + right) // 2
        if artiste[mid] == target:
            return "found it!"
        elif artiste[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return "Not found!"
print(binary_search3(artiste, "Bassey"))                    


# Binary search 4
artiste2 = ["wizkid", "davido", "Bassey", "chioma", "sean tizzle"] 
def list_search(find_artiste, target):
    for find_artiste in artiste2:
        if find_artiste == target:
            return "found him!"
    return "not available"

print(list_search(artiste2, "sean tizzle"))           