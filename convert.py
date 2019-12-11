import numpy as np

# A dictionary of all symbols that are currently supported mapped to their LaTeX representation
symbol_library = {
    "-": "-",
    "(": "(",
    ")": ")",
    "+": "+",
    "div": "\\frac{}{}",
    "gt": ">",
    "lt": "<",
    "sum": "\sum",
    "times": "\\times",
    "u": "u",
    "v": "v",
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "=": "=",
    "leq": "\leq",
    "geq": "\geq"
}

def is_left(box1, box2):
    """
    Takes in two bounding boxes and returns true if box1 is further left than box2.
    """
    return get_center(box1)[0] < get_center(box2)[0]

def is_below(box1, box2):
    """
    Takes in two bounding boxes and returns true if box1 is below box2.
    """
    return get_center(box1)[1] < get_center(box2)[1]

def close_horizontally(box1, box2, threshold):
    """
    Returns true if the center of box1 is less than threshold pixels away horizontally from the
    center of box2.
    """
    return abs(get_center(box1)[0] - get_center(box2)[0]) <= threshold

def close_vertically(box1, box2, threshold):
    """
    Returns true if the center of box1 is less than threshold pixels away vertically from the
    center of box2.
    """
    return abs(get_center(box1)[1] - get_center(box2)[1]) <= threshold

def get_center(box):
    """
    Returns the location of the center of the bounding box.
    """
    left = box[0][0]
    right = box[1][0]
    top = box[2][1]
    bottom = box[0][1]

    return [left + abs(right - left) / 2, bottom + abs(top - bottom) / 2]

def combine_boxes(corners):
    """
    Given a list of bounding boxes, combine their coordinates to create a large bounding box that
    contains all of them.
    """
    left_limit = np.infty  # The location of the furthest left pixel
    right_limit = -np.infty  # The location of the furthest right pixel
    upper_limit = -np.infty  # The location of the highest pixel
    lower_limit = np.infty  # The location of the lowest pixel

    for box in corners:
        left = box[0][0]
        right = box[1][0]
        top = box[2][1]
        bottom = box[0][1]

        if left < left_limit:
            left_limit = left
        if right > right_limit:
            right_limit = right
        if top > upper_limit:
            upper_limit = top
        if bottom < lower_limit:
            lower_limit = bottom

    new_box = [[left_limit, lower_limit], [right_limit, lower_limit], [right_limit, upper_limit], [left_limit, upper_limit]]
    return new_box

def fix_parenthesis(output, corners):
    """
    Takes in a list of model outputs. If there are an odd number of parenthesis it converts the odd
    parenthesis into the models second most likely prediction. Currently only able to handle one
    set of parenthesis.
    """
    left_count = 0
    right_count = 0

    for out in output:
        if out[0] == "(":
            left_count += 1
        if out[0] == ")":
            right_count += 1

    if left_count == right_count:
        return output, corners

    else:
        new_output = []
        if abs(left_count - right_count) == 1: # If there is only one parenthesis in the equation
            for out in output:
                if out[0] == "(" or out[0] == ")":
                    swap = [out[1], out[0]] # Use the models second most likely output instead
                    new_output.append(swap)
                else:
                    new_output.append(out)

        return new_output, corners

def find_equals(output, corners):
    """
    Our model was not trained on the equal to symbol. We must detect it by finding two minus
    signs that are close to each other.
    """
    new_output = []
    new_corners = []
    indices = [] # A list of indices of output that have already been checked to see if they are in an equals sign
    vertical_threshold = 40
    horizontal_threshold = 40

    for i in range(len(output)):
        is_equals = False
        for j in range(len(output)):
            if i != j and i not in indices and j not in indices:
                symbol1 = output[i][0]
                symbol2 = output[j][0]
                if symbol1 == "-" and symbol2 == "-" and close_vertically(corners[i], corners[j], vertical_threshold) and close_horizontally(corners[i], corners[j], horizontal_threshold):
                    is_equals = True
                    equals = ["=", "="]
                    box = combine_boxes([corners[i], corners[j]])
                    new_output.append(equals)
                    new_corners.append(box)
                    indices.append(i)
                    indices.append(j)

        if not is_equals and i not in indices:
            new_output.append(output[i])
            new_corners.append(corners[i])

    return new_output, new_corners

def find_leq(output, corners):
    """
    Our model was not trained on the less than or equal to symbol. We must detect it by finding
    two minus signs that are close to each other.
    """
    new_output = []
    new_corners = []
    indices = [] # A list of indices of output that have already been checked to see if they are in an equals sign
    vertical_threshold = 40
    horizontal_threshold = 40

    for i in range(len(output)):
        is_leq = False
        for j in range(len(output)):
            if i != j and i not in indices and j not in indices:
                symbol1 = output[i][0]
                symbol2 = output[j][0]
                if symbol1 == "lt" and symbol2 == "-" and is_below(corners[2], corners[1]) and close_vertically(corners[i], corners[j], vertical_threshold) and close_horizontally(corners[i], corners[j], horizontal_threshold):
                    is_leq = True
                    leq = ["leq", "leq"]
                    box = combine_boxes([corners[i], corners[j]])
                    new_output.append(leq)
                    new_corners.append(box)
                    indices.append(i)
                    indices.append(j)

        if not is_leq and i not in indices:
            new_output.append(output[i])
            new_corners.append(corners[i])

    return new_output, new_corners

def find_geq(output, corners):
    """
    Our model was not trained on the greater than or equal to symbol. We must detect it by finding
    two minus signs that are close to each other.
    """
    new_output = []
    new_corners = []
    indices = [] # A list of indices of output that have already been checked to see if they are in an equals sign
    vertical_threshold = 40
    horizontal_threshold = 40

    for i in range(len(output)):
        is_geq = False
        for j in range(len(output)):
            if i != j and i not in indices and j not in indices:
                symbol1 = output[i][0]
                symbol2 = output[j][0]
                if symbol1 == "gt" and symbol2 == "-" and is_below(corners[2], corners[1]) and close_vertically(corners[i], corners[j], vertical_threshold) and close_horizontally(corners[i], corners[j], horizontal_threshold):
                    is_geq = True
                    geq = ["geq", "geq"]
                    box = combine_boxes([corners[i], corners[j]])
                    new_output.append(geq)
                    new_corners.append(box)
                    indices.append(i)
                    indices.append(j)

        if not is_geq and i not in indices:
            new_output.append(output[i])
            new_corners.append(corners[i])

    return new_output, new_corners

def find_div(output, corners):
    """
    Our model was not trained on fractional division symbols (numerator and denominator physically
    located over top of each other). We must detect it by finding a minus sign with at least one
    symbol above and below it.
    """
    new_output = []
    new_corners = []
    indices = []  # A list of indices of output that have already been checked to see if they are in an equals sign

    for i in range(len(output)):
        symbol1 = output[i][0]
        box1 = corners[i]
        if i not in indices and symbol1 == "-":
            num_above = 0  # The number of symbols above the minus sign
            num_below = 0  # The number of symbols below the minus sign
            left_limit = box1[0][0]
            right_limit = box1[1][0]

            for j in range(len(output)): # Loop through all other symbols
                if i != j and i not in indices and j not in indices:
                    box2 = corners[j]
                    center = get_center(box2)[0]
                    if center >= left_limit and center <= right_limit and is_below(box1, box2): # If the symbol at j is above the minus sign and its center is within the left and right limits of the minus sign
                        num_above += 1
                    elif center >= left_limit and center <= right_limit and is_below(box2, box1): # If the symbol at j is below the minus sign and its center is within the left and right limits of the minus sign
                        num_below += 1

            if num_above > 0 and num_below > 0:
                div = ["div", "div"]
                new_output.append(div)
                new_corners.append(corners[i]) # It maintains its bounding box, only the symbol type changes
                indices.append(i)

            else:
                new_output.append(output[i])
                new_corners.append(corners[i])

        else:
            new_output.append(output[i])
            new_corners.append(corners[i])

    return new_output, new_corners

def to_latex(output, corners):
    """
    Takes in a list of model outputs, the bounding boxes for each character in the input image,
    and the area of the bounding box. Returns a properly formatted LaTeX mathermatical expression.
    """
    left_limit = np.infty # The location of the furthest left pixel
    right_limit = -np.infty # The location of the furthest right pixel
    upper_limit = -np.infty # The location of the highest pixel
    lower_limit = np.infty # The location of the lowest pixel

    for box in corners:
        left = box[0][0]
        right = box[1][0]
        top = box[2][1]
        bottom = box[0][1]

        if left < left_limit:
            left_limit = left
        if right > right_limit:
            right_limit = right
        if top > upper_limit:
            upper_limit = top
        if bottom < lower_limit:
            lower_limit = bottom

    output = [['2', '3'], ['-', '9'], ['-', '9'], ['u', '8'], ['div', 'div'], ['+', '9'], ['-', '9'], ['2', '3'],
              ['7', '8'], ['0', '2'], ['1', '1'], ['0', '2'], ['5', '9'], ['(', '7'], [')', '3'], ['u', '8'],
              ['3', '3']]

    corners = [[[315, 390], [460, 390], [460, 539], [315, 539]], [[1464, 331], [1554, 331], [1554, 354], [1464, 354]],
               [[1929, 320], [2000, 320], [2000, 342], [1929, 342]],
               [[1313, 298], [1409, 298], [1409, 390], [1313, 390]], [[133, 295], [577, 295], [577, 351], [133, 351]],
               [[703, 287], [798, 287], [798, 378], [703, 378]], [[1936, 284], [2029, 284], [2029, 311], [1936, 311]],
               [[1191, 274], [1284, 274], [1284, 401], [1191, 401]],
               [[1586, 255], [1697, 255], [1697, 402], [1586, 402]],
               [[2183, 246], [2284, 246], [2284, 343], [2183, 343]],
               [[2105, 245], [2148, 245], [2148, 358], [2105, 358]],
               [[2297, 239], [2412, 239], [2412, 357], [2297, 357]], [[909, 233], [1012, 233], [1012, 391], [909, 391]],
               [[1086, 220], [1198, 220], [1198, 435], [1086, 435]],
               [[1698, 212], [1820, 212], [1820, 451], [1698, 451]], [[310, 163], [425, 163], [425, 286], [310, 286]],
               [[165, 106], [278, 106], [278, 281], [165, 281]]]

    output, corners = fix_parenthesis(output, corners)
    output, corners = find_equals(output, corners)
    output, corners = find_geq(output, corners)
    output, corners = find_leq(output, corners)
    output, corners = find_div(output, corners)

    equation = []
    center_loc = [] # A list of the center of the bounding boxes for each symbol or sub-equation
    removed = [] # A list of indices that have been removed

    for i in range(len(output)):
        symbol1 = output[i][0]
        box1 = corners[i]
        if i not in removed and symbol1 == "div":
            above_idx = [] # The index of all symbols in output that are in the numerator
            below_idx = [] # The index of all symbols in output that are in the denominator

            left_limit = box1[0][0] # The furthest left point of the division sign
            right_limit = box1[1][0] # The furthest right point of the division sign

            for j in range(len(output)):
                if j != i and j not in removed:
                    box2 = corners[j]
                    if is_below(box2, box1) and get_center(box2)[0] >= left_limit and get_center(box2)[0] <= right_limit:
                        above_idx.append(j)
                    if is_below(box1, box2) and get_center(box2)[0] >= left_limit and get_center(box2)[0] <= right_limit:
                        below_idx.append(j)

            new_box = combine_boxes(corners[j] for j in above_idx + below_idx) # A bounding box that fits over the entire div equation
            new_center = get_center(new_box)
            center_loc.append(new_center)

            div_as_symbol = symbol_library["div"]
            numerator = []
            denominator = []

            for j in sorted(above_idx, key = lambda x: get_center(corners[x])[0]): # List the elements of the numerator from left to right
                numerator.append(symbol_library[output[j][0]])

            for j in sorted(below_idx, key=lambda x: get_center(corners[x])[0]): # List the elements in the denominator from left to right
                denominator.append(symbol_library[output[j][0]])

            equation.append(div_as_symbol[:6] + "".join(numerator) + div_as_symbol[6:8] + "".join(denominator) + div_as_symbol[8:])

            removed.append(i)
            for j in above_idx + below_idx:
                removed.append(j)

    for i in range(len(output)): # Loop through all symbols that are left after removing division symbols
        if i not in removed:
            equation.append(symbol_library[output[i][0]])
            center_loc.append(get_center(corners[i]))

    equation = [i for i, _ in sorted(zip(equation, center_loc), key=lambda x: x[1][0])]

    return "$" + "".join(equation) + "$"