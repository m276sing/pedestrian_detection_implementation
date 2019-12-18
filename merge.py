def merge_sort(input_list):
    element = 1
    while element <= len(input_list):
        for iterate in range(0, len(input_list), element * 2):
            i, j = iterate, min(len(input_list), iterate + 2 * element)
            middle = iterate + element
            a, b = i, middle
            while a < middle and b < j:
                if not input_list[a] > input_list[b]:
                    a += 1
                else:
                    temporary = input_list[b]
                    input_list[a + 1: b + 1] = input_list[a:b]
                    input_list[a] = temporary
                    a, mid, b = a + 1, middle + 1, b + 1
        element *= 2
    flag = 0
    for x in range(0, len(input_list) - 1):
        if input_list[x] == input_list[x + 1]:
            flag = 1
    if flag == 1:
        print("False")
    else:
        print("True")


def final():
    merge_sort([7, 3, 1, 5, 10])



if __name__ == "__main__":
    final()
