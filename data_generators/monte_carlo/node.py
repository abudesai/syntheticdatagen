
import numpy as np
import hashlib
from tree_traverse import TreeTraverser

class Node:     
    def __init__(self, tree_type, level, node_num, parent_node, parent_id):
        self.type = tree_type    # org / activity
        self.level = level
        self.node_num = node_num
        self.parent_node = parent_node
        self.parent_id = parent_id
        self.children = []
        self.properties = {}
        if self.type == 'Org':
            salt = str(np.random.randint(0, 1e8))
        else: salt = ''
        id_str = f'{self.type}_{self.level}_{self.node_num}_{salt}_'
        parent_str = self.parent_node.id if self.parent_node is not None else 'None'
        hash_str = id_str + parent_str
        self.id = id_str +  str(int(hashlib.sha256(hash_str.encode('utf-8')).hexdigest(), 16) % 10**5 )
            

    def add_children(self, num_children_bounds):
        if not self.children: 
            lb, ub = num_children_bounds
            num_children = np.random.randint(low=lb, high=ub+1)
            for i in range(num_children):
                self.children.append( Node( tree_type= self.type, 
                    level=self.level+1, 
                    node_num=i, 
                    parent_node=self,
                    parent_id=self.id ) )        
        else:
            for child in self.children:
                child.add_children(num_children_bounds)
                

    def get_node_by_id_from_tree(self, node_id):
        if self.id == node_id: return self
        for node in TreeTraverser.traverse_pre_order(self):
            if node.id == node_id: return node
        return None


    def get_flat_hierarchy(self):
        all_hierarchy = []
        for node in TreeTraverser.traverse_pre_order_leaves_only(self):
            curr = node
            node_org_hiers = [curr.id]
            while curr.parent_id is not None:
                node_org_hiers.insert(0, curr.parent_id)
                curr = curr.parent_node
            all_hierarchy.append(node_org_hiers)
        return all_hierarchy


    @property
    def leaf_count(self):
        leaf_count =0
        for node in TreeTraverser.traverse_pre_order(self):
            if len(node.children) == 0:
                leaf_count += 1
        return leaf_count

    @property
    def node_count(self):
        node_count =0
        for node in TreeTraverser.traverse_pre_order(self): node_count += 1
        return node_count


    @property
    def info(self):
        return f"{self.type} - (L:{self.level}|N:{self.node_num}|C:{len(self.children)}|Uid:{self.id})"

    @property
    def is_leaf(self):
        return len(self.children) == 0
   
    def __str__(self):
        return " " * 4 * self.level + self.info

    
   
class Tree(Node): 
    def __init__(self, tree_type):
        super().__init__(tree_type, 0, 0, None, None)

    def create_tree(self, num_children_per_gen = []):
        for num_children_bounds in num_children_per_gen:            
            self.add_children(num_children_bounds)


class OrgTree(Tree):
    def __init__(self):
        super().__init__("Org")
        

class ActivityTree(Tree):
    def __init__(self):
        super().__init__("Act")


if __name__ == "__main__":
    tree = OrgTree()
    tree.create_tree(num_children_per_gen=[(2,2)])

    for node in TreeTraverser.traverse_pre_order(tree):
        print(node)

    # print(f"tree.leaf_count: {tree.leaf_count}")

    # gen = TreeTraverser.traverse_pre_order(tree)
    # print(next(gen))

    # print(tree.get_flat_hierarchy())

    # for node in TreeTraverser.traverse_pre_order_leaves_only(tree):
    #     print(node)

