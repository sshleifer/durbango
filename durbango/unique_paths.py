def how_many_ways(grid, row, col):
    if grid[row][col] == 1:
        return 0
    if row == 0 and col == 0:
        return 1
    from_above = 0
    from_side = 0
    if row > 0:
        from_above = how_many_ways(grid, row - 1, col)
    if col > 0:
        from_side = how_many_ways(grid, row, col - 1)
    return from_above + from_side


def max_heapify(arr: list, i: int) -> list:
    """Max heapify the subtree rooted at Arr[i]. Log complexity."""
    N = len(arr)
    left =  2 * i +1
    right = left+1
    val = arr[i]
    largest = i
    if left < N and arr[left] > val:
        largest = left
    if right < N and arr[right] > val:
        largest = right
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # swap parent and child
        max_heapify(arr, largest)



def build_max_heap(A):
    for i in range(len(A) // 2, -1, -1):
        max_heapify(A, i)


def is_bipartite(edges):
    edges = list(sorted([(i, set(x)) for i,x in enumerate(edges)], key=lambda x: -len(x[1])))
    src, friends = edges.pop(0)
    l = {src}
    r = friends
    seen = set()
    while edges:
        i, out_friends = edges.pop(0)
        should_be_l = i in l or r.intersection(out_friends)
        should_be_r = i in r or l.intersection(out_friends)
        if should_be_l and should_be_r: return False
        elif should_be_l:  # add i to left, friends to right
            l.add(i)
            r |= out_friends
        elif should_be_r: # add i to right, friends to left
            r.add(i)
            l |= out_friends
        else:  # can be either, process later
            if i in seen:
                pass
            else:
                seen.add(i)
                edges.append((i, out_friends))
        i += 1
    return True

