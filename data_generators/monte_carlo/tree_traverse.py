

class TreeTraverser:
    @staticmethod
    def traverse_pre_order(root):
        def traverse(current):
            yield current
            for child in current.children:
                for node in traverse(child):
                    yield node        
        for node in traverse(root):
            yield node

    @staticmethod
    def traverse_pre_order_leaves_only(root):
        def traverse(current):
            if current.is_leaf: yield current
            for child in current.children:
                for node in traverse(child):
                    if node.is_leaf: yield node        
        for node in traverse(root):
            yield node

    @staticmethod
    def traverse_post_order(root):
        def traverse(current):
            for child in current.children:
                for node in traverse(child):
                    yield node   
            yield current     
        for node in traverse(root):
            yield node


