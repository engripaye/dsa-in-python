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
toys = ["car", "doll", "car", "train"]
def find_toys(toys, target):
    for toy in toys:
        if toy == toys:
            return "Found it!"
        return "not here!"
    print(find_toys(toys, "ball"))


# Linear search 2
toys = ["aeroplane", "jet", "helo", "cop"]
def search_object(toys, target):
    for object in toys:
        if object == target:
            return "Found it!"
        return "Not available!"
    print(search_object(toys, "cop"))       


# Linear Search 3
data = ["players", "coach", "board-members", "asst-coach"]
def search_data(data, target):
    for object in data:
        if object == target:
            return "found it!"
        return "not available!"
    print(search_data(data, "asst-coach"))


# Linear Search 4
data = ["biro", "book", "ruler", "math-set"]
def search_schl_tool(tools, target):
    for tools in data:
        if tools == target:
            return "Found it!"
        return "not available!"
    print(search_schl_tool(data, "book"))


# Binary Search 1
list = ["doll", "plane", "bus", "van"]
def binary_search(toys, target):
    left, right = 0 , len(list) - 1
    while left <= right:
        mid = (left + right) // 2
        if toys[mid] == target:
            return "Found it!"
        elif toys[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
        return "Not found!"
    print(binary_search(list, "bus"))      

# Binary search 2
music = ["bounce", "fly", "daddy yo", "only God"]
def binary_search2(findMusic, target):
    left, right = 0 , len(music) - 1
    while left <= right:
        mid = (left + right) // 2
        if findMusic[mid] == target:
            return "Found it!"
        elif findMusic[mid] < target:
            left = mid + 1
        else:
            right = mid + 1
        return "Not found"                     
    print(binary_search2(music, "bounce"))        

# Binary search 3
artiste = ["wizkid", "davido", "Bassey", "Chioma"]
def binary_search3(findArtiste, target):
    left, right = 0, len(artiste) - 1
    while left <= right:
        mid = (left + right) // 2
        if findArtiste[mid] == target:
            return "found it!"
        elif findArtiste[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
        return "Not found!"
    print(binary_search3(artiste, "wizkid"))                    