import numpy as np


def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient


def fast_non_dominated_sort(objectives):


    if objectives is None or len(objectives) == 0:
        return []

    n = objectives.shape[0]
    if n == 0:
        return []


    S = [[] for _ in range(n)]
    n_p = np.zeros(n, dtype=int)
    rank = np.zeros(n, dtype=int)
    fronts = [[]]


    for i in range(n):
        for j in range(i + 1, n):
            try:
                if dominates(objectives[i], objectives[j]):
                    S[i].append(j)
                    n_p[j] += 1
                elif dominates(objectives[j], objectives[i]):
                    S[j].append(i)
                    n_p[i] += 1
            except Exception as e:
                print(f"fail: {e}")
                continue


        if n_p[i] == 0:
            rank[i] = 0
            fronts[0].append(i)

    if not fronts[0]:
        return []

    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n_p[q] -= 1
                if n_p[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        if Q:
            fronts.append(Q)
        else:
            break

    return fronts

def dominates(a, b):

    not_worse = np.all(a >= b)

    better = np.any(a > b)

    return not_worse and better


def crowding_distance(objectives, front):

    n = len(front)
    distances = np.zeros(n)

    if n == 0:
        return distances

    m = objectives.shape[1]

    for obj_idx in range(m):

        sorted_indices = np.argsort(objectives[front, obj_idx])
        sorted_front = [front[i] for i in sorted_indices]


        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')


        if n > 2:
            min_val = objectives[sorted_front[0], obj_idx]
            max_val = objectives[sorted_front[-1], obj_idx]

            if abs(max_val - min_val) < 1e-10:
                continue

            norm = max_val - min_val
            for i in range(1, n - 1):
                prev_val = objectives[sorted_front[i - 1], obj_idx]
                next_val = objectives[sorted_front[i + 1], obj_idx]
                distances[sorted_indices[i]] += (next_val - prev_val) / norm

    return distances



def get_pareto_front(objectives):

    fronts = fast_non_dominated_sort(objectives)
    return fronts[0] if fronts else []


# Test
if __name__ == "__main__":
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    ########### 2D example ######################
    a = np.random.rand(20, 2)
    is_efficient = is_pareto_efficient_simple(a)

    # 测试快速非支配排序
    fronts = fast_non_dominated_sort(a)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(a[:, 0], a[:, 1], 'bo')
    for i in range(20):
        if is_efficient[i]:
            plt.plot(a[i, 0], a[i, 1], 'ro')
    plt.title('is_pareto_efficient_simple')

    plt.subplot(1, 2, 2)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, front in enumerate(fronts):
        if i < len(colors):
            color = colors[i]
        else:
            color = 'gray'
        front_points = a[front]
        plt.plot(front_points[:, 0], front_points[:, 1], 'o', color=color, label=f'Front {i}')
    plt.title('fast_non_dominated_sort')
    plt.legend()

    plt.tight_layout()
    plt.show()

    ########### 3D example ######################
    a = np.random.rand(200, 3)
    fronts = fast_non_dominated_sort(a)

    fig = plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")

    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, front in enumerate(fronts[:3]):  # 只显示前3个前沿
        if i < len(colors):
            color = colors[i]
        else:
            color = 'gray'
        front_points = a[front]
        ax.scatter3D(front_points[:, 0], front_points[:, 1], front_points[:, 2], color=color, label=f'Front {i}')

    ax.legend()
    plt.title('3D Pareto Fronts')
    plt.show()