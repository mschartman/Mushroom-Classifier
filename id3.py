#!/usr/bin/env python
from math import log
import random
import time

DATASET = 'agaricus-lepiota.data'
ATTRIBUTES = 'agaricus-lepiota.names'

g_attributes = [] # Doesn't include poisonous or edible column
g_attributes_dictionary = {}

class Node(object):

    def __init__(self, children=None, attr=None):
        if children is None:
            children = []
        self.children = children
        self.attribute = attr
    def __str__(self):
        return str(self.attribute)

def print_tree(root):
    print ''
    print root.attribute
    for i in root.children:
        print '\t\t%s' % i.attribute
        for j in i.children:
            print '\t\t\t\t%s' % j.attribute
            for k in j.children:
                print '\t\t\t\t\t\t%s' % k.attribute
                for l in k.children:
                    print '\t\t\t\t\t\t\t\t%s' % l.attribute
                    for m in l.children:
                        print '\t\t\t\t\t\t\t\t\t\t%s' % m.attribute

def parse_attributes():
    with open(ATTRIBUTES, 'r+') as attributes_file:
        for line in attributes_file:
            pair = line.strip().split()
            g_attributes.append(pair[0])
            g_attributes_dictionary[pair[0]] = pair[1].split(',')

def prepare_datasets():
    positive_dataset = []
    negative_dataset = []

    with open(DATASET, 'r+') as dataset_file:
        for line in dataset_file:
            attributes = line.split(',')

            # Get rid of newline character on last attribute
            attributes[-1] = attributes[-1].strip()

            if attributes[0] == 'e':
                positive_dataset.append(('e+', attributes[1:]))
            else:
                negative_dataset.append(('p-', attributes[1:]))

    training_data = []
    test_data = []

    while positive_dataset and negative_dataset:
        rcp = random.choice(positive_dataset)
        rcn = random.choice(negative_dataset)
        positive_dataset.remove(rcp)
        negative_dataset.remove(rcn)
        training_data.append(rcp)
        training_data.append(rcn)

        if not positive_dataset or not negative_dataset:
            break

        rcp = random.choice(positive_dataset)
        rcn = random.choice(negative_dataset)
        positive_dataset.remove(rcp)
        negative_dataset.remove(rcn)
        training_data.append(rcp)
        training_data.append(rcn)

        if not positive_dataset or not negative_dataset:
            break

        rcp = random.choice(positive_dataset)
        rcn = random.choice(negative_dataset)
        positive_dataset.remove(rcp)
        negative_dataset.remove(rcn)
        training_data.append(rcp)
        training_data.append(rcn)

        if not positive_dataset or not negative_dataset:
            break

        rcp = random.choice(positive_dataset)
        rcn = random.choice(negative_dataset)
        positive_dataset.remove(rcp)
        negative_dataset.remove(rcn)
        test_data.append(rcp)
        test_data.append(rcn)

    return training_data, test_data

# Entropy is used to determine which node to split next in the algorithm,
# the higher the entropy, the higher the potential to improve the classification here
def entropy(examples):
    if len(examples) == 0:
        return 0

    positive_examples = [i for i in examples if i == 'e+']
    negative_examples = [i for i in examples if i == 'p-']

    if len(positive_examples) == 0 or len(negative_examples) == 0:
        return 0

    prob_pos = float(len(positive_examples))/float(len(examples))
    prob_neg = float(len(negative_examples))/float(len(examples))

    return (-1.0) * prob_pos * log(prob_pos, 2.0) - prob_neg * log(prob_neg, 2.0)

# Gain is computed to estimate the gain produced by a split over an attribute
def gain(examples, attribute):
    if len(examples) == 0:
        return 0

    attr_values = g_attributes_dictionary[attribute]

    sum = 0.0
    examples_with_attr = []

    for value in attr_values:
        for entry in examples:
            if entry[1][g_attributes.index(attribute)] == value :
                examples_with_attr.append(entry[0])
        entropy_examples_with_attr = entropy(examples_with_attr)
        if entropy_examples_with_attr == 0:
            return 0
        else:
            sum += float(len(examples_with_attr)) / float(len(examples) * float(entropy_examples_with_attr))

    return entropy([i[0] for i in examples]) - sum

def ID3(examples, target_attribute, attributes):
    # Create a root node for the tree
    root = Node()

    # If all examples are positive, Return the single-node tree Root, with label = e
    if len([ex for ex in examples if ex[0] == 'p-']) == 0:
        root.attribute = 'e+'
        return root

    # If all examples are negative, Return the single-node tree Root, with label = p
    if len([ex for ex in examples if ex[0] == 'e+']) == 0:
        root.attribute = 'p-'
        return root

    # If number of predicting attributes is empty, then Return the single node tree Root,
    # with label = most common value of the target attribute in the examples
    if len(attributes) == 0:
        pos_count = 0
        neg_count = 0

        for ex in examples:
            if ex[0] == 'e+':
                pos_count += 1
            else:
                neg_count += 1

        if pos_count >= neg_count:
            root.attribute = 'e+'
        else:
            root.attribute = 'p-'

        return root

    else:
        # The Attribute that best classifies examples
        best_attr = (None, 0.0)
        for attr in attributes:
            attr_gain = gain(examples, attr)
            if attr_gain >= best_attr[1]:
                best_attr = (attr, attr_gain)
        a = best_attr[0]

        # Decision Tree attribute for Root = A
        root.attribute = a

        index_of_a = g_attributes.index(a)
        possibles_values_of_a = g_attributes_dictionary[a]

        # For each possible value of A
        for val in possibles_values_of_a:
            # Add a new tree branch below Root, corresponding to the test A = val
            new_node = Node(attr=val)
            root.children.append(new_node)
            child = new_node

            # Let Examples(val) be the subset of examples that have the value val for A
            ex_with_val_for_a = []
            for ex in examples:
                if ex[1][index_of_a] == val:
                    ex_with_val_for_a.append(ex)

            # If Examples(val) is empty
            if len(ex_with_val_for_a) == 0:
                pos_count = 0
                neg_count = 0

                for ex in examples:
                    if ex[0] == 'e+':
                        pos_count += 1
                    else:
                        neg_count += 1

                leaf_node = Node()

                if pos_count >= neg_count:
                    leaf_node.attribute = 'e+'
                else:
                    leaf_node.attribute = 'p-'

                child.children.append(leaf_node)

            # Else below this new branch add the subtree for running ID3 without A
            else:
                attributes_copy = attributes[:]
                attributes_copy.remove(a)
                child.children.append(ID3(ex_with_val_for_a, target_attribute, attributes_copy))
        return root

def classify(example, root):
    if len(root.attribute) > 2:
        current_attr_location = g_attributes.index(root.attribute)
        for child in root.children:
            if child.attribute == example[1][current_attr_location]:
                return classify(example, child)
    elif root.attribute == 'e+':
        return 'e+'
    elif root.attribute == 'p-':
        return 'p-'
    else:
        return classify(example, root.children[0])

if __name__ == '__main__':
    start = time.time()
    training_data, test_data = prepare_datasets()
    parse_attributes()
    tree = ID3(training_data, 'e+', g_attributes[:])
    print_tree(tree)

    correct = 0
    incorrect = 0
    print '--------'
    for example in test_data:
        result = classify(example, tree)
        print result, example[0]
        if result == example[0]:
            correct += 1
        else:
            incorrect += 1
    percent_correct = 100.0 * float(correct)/(float(correct) + float(incorrect))
    run_time = time.time() - start
    print '%s correct, %s incorrect' % (correct, incorrect)
    print 'Percent correct: %s' % percent_correct
    print 'Runtime: %s' % run_time
