# Fill array to certain size with existing values in order to keep consistent input size
def FillArrayToSize(arr, size):

    if len(arr) < size:
        numRepetitions = int(size / len(arr)) + 1
        arr *= numRepetitions
    
    arr = arr[0:size]
    return arr