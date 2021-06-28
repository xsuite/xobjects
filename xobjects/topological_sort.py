from collections import defaultdict


def topological_sort(source):
    """Sort tasks defined as {child: parents} in a list"""

    # Store parent count and parent dictionary.
    graph = {}
    num_parents = defaultdict(int)
    for child, parents in source.items():
        for parent in parents:
            graph.setdefault(parent, []).append(child)
            num_parents[child] += 1

    # Begin with the parent-less items.
    result = [child for child, parents in source.items() if len(parents) == 0]
    result.extend([item for item in graph if num_parents[item] == 0])

    # Descend through graph, removing parents as we go.
    for parent in result:
        if parent in graph:
            for child in graph[parent]:
                num_parents[child] -= 1
                if num_parents[child] == 0:
                    result.append(child)
            del graph[parent]

    # If there's a cycle, just throw in whatever is left over.
    has_cycle = bool(graph)
    if has_cycle:
        result.extend(list(graph.keys()))
    return result, has_cycle


def topological_sort_par(source):
    """Sort tasks defined as {child: parents} in a list of parallelizable list of tasks"""

    # Store parent count and parent dictionary.
    graph = {}
    num_parents = defaultdict(int)
    for child, parents in source.items():
        for parent in parents:
            graph.setdefault(parent, []).append(child)
            num_parents[child] += 1

    # Begin with the parent-less items.
    result = [child for child, parents in source.items() if len(parents) == 0]
    result.extend([item for item in graph if num_parents[item] == 0])

    # Descend through graph, removing parents as we go.
    result2 = [result.copy()]
    curr = []
    for parent in result:
        if parent in graph:
            for child in graph[parent]:
                num_parents[child] -= 1
                if num_parents[child] == 0:
                    result.append(child)
                    curr.append(child)
            del graph[parent]
            if len(curr) > 0:
                result2.append(curr)
            curr = []

    if len(curr) > 0:
        result.append(curr)

    has_cycle = bool(graph)
    if has_cycle:
        result.append(list(graph.keys()))
    return result2, has_cycle
