class Node:
    def __init__(self, data):
        self.val = data
        self.left = None
        self.right = None

def make_tree():
    root = Node(1);
    root.left = Node(2);
    root.right = Node(3);
    root.left.left = Node(4);
    root.left.right = Node(5);
    root.right.right = Node(6);
    return root

def traverse_level_order(root):
    # Base case
    if root is None:
        return
    q = [root]
    result = []
    while q:
        # Dequeue all nodes of current level and  enqueue all nodes of next level
        temp = q.pop(0)
        print(temp.val, end=' ')
        if temp.left:
            q.append(temp.left)
        if temp.right:
            q.append(temp.right)



