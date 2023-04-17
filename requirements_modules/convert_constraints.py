import os
readable_constraints_file = 'constraints/readable_constraints.txt'
label_order = ['Ped', 'Car', 'Mobike', 'SmalVeh', 'MedVeh', 'LarVeh', 'Bus', 'EmVeh'] + \
              ['MovAway', 'MovTow', 'Mov', 'Rev', 'Brake', 'Stop', 'IncatLft', 'IncatRht', 'HazLit', 'TurLft',
               'TurRht', 'MovRht', 'MovLft', 'Ovtak', 'Wait2X', 'XingFmLft', 'XingFmRht', 'Xing', 'PushObj'] + \
              ['VehLane', 'OutgoLane', 'OutgoCycLane', 'OutgoBusLane', 'IncomLane', 'IncomCycLane', 'IncomBusLane',
               'Pav', 'LftPav', 'RhtPav', 'Jun', 'xing', 'BusStop', 'parking', 'LftParking', 'rightParking']
indexed_label_order = {elem:i for i,elem in enumerate(label_order)}
print(label_order, indexed_label_order, len(label_order))

def get_constraints_from_file():
    all_labels = set([])
    all_constraints = []
    with open(readable_constraints_file) as f:
        for line in f:
            head, body = line.strip().split(' :- ')

            head_literal = head.split(' ')
            if 'neg ' in head:
                all_labels.add(head_literal[1])
                head_literal = ('neg', head_literal[1])
            else:
                all_labels.add(head_literal[0])
                head_literal = ('pos', head_literal[0])


            body = body.split(' ')
            body_literals = []
            i = 0
            while i < len(body):
                if body[i] == 'neg':
                    literal = body[i+1]
                    body_literals.append(('neg', literal))
                    i += 2
                else:
                    literal = body[i]
                    body_literals.append(('pos', literal))
                    i += 1
                all_labels.add(literal)
            all_constraints.append((head_literal, body_literals))
            # print(head_literal)
            # print(body_literals)

        all_labels = list(all_labels)
        print(all_labels, len(all_labels))
        return all_labels, all_constraints

def set_difference(set1, set2):
    diff = []
    for elem in set1:
        if elem not in set2:
            diff.append(elem)
    return diff


def is_valid_line(line, labels_to_exclude):
    for label in labels_to_exclude:
        if label in line:
            return False
    return True


def make_indexed_constraints_set(indexed_label_order, readable_constraints):
    s = ''
    indexed_constraints = []
    for (read_head, read_body) in readable_constraints:
        s += '0.0 '
        if read_head[1] not in indexed_label_order:
            indexed_label_order[read_head[1]] = len(indexed_label_order)

        head = str(indexed_label_order[read_head[1]]) if read_head[0] == 'pos' else 'n' + str(indexed_label_order[read_head[1]])
        s += head + ' :- '
        for elem in read_body:
            if elem[1] not in indexed_label_order:
                indexed_label_order[elem[1]] = len(indexed_label_order)
            s += str(indexed_label_order[elem[1]]) if elem[0] == 'pos' else 'n' + str(indexed_label_order[elem[1]])
            s += ' '
        s += '\n'
    with open('constraints/roadpp_indexed_constraints.txt', 'w') as g:
        g.write(s)
    print(len(indexed_label_order))

all_labels, all_constraints = get_constraints_from_file()

labels_to_remove = set_difference(all_labels, label_order)
new_roadpp_labels = set_difference(label_order, all_labels)
print(new_roadpp_labels)
print(labels_to_remove)
indexed_constraints_set = make_indexed_constraints_set(indexed_label_order, all_constraints)
