import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import random
from matplotlib.colors import Normalize
from matplotlib import cm


def compute_graph_metrics(W):
    """
    주어진 weight matrix에 대해 connectivity variance, mean connectivity, second eigenvalue를 계산합니다.

    Parameters:
    W (numpy.ndarray): n x n 정방 weight matrix

    Returns:
    dict: 다음 메트릭들을 포함하는 딕셔너리
        - connectivity_variance: 전체 그래프의 connectivity variance
        - mean_connectivity: 전체 그래프의 mean connectivity
        - second_eigenvalue: 라플라시안 행렬의 두번째로 작은 고유값
    """
    # 기본 검증
    if not isinstance(W, np.ndarray):
        W = np.array(W)

    if W.shape[0] != W.shape[1]:
        raise ValueError("Weight matrix must be square")

    n = W.shape[0]  # 정점 수

    # 각 정점의 degree 계산
    degrees = np.sum(W, axis=1)

    # 각 정점의 connectivity variance 계산
    vertex_var = np.zeros(n)
    for i in range(n):
        avg_degree_i = degrees[i] / n
        vertex_var[i] = np.mean((W[i, :] - avg_degree_i) ** 2)

    # 전체 그래프의 connectivity variance
    connectivity_variance = np.mean(vertex_var)

    # 전체 그래프의 mean connectivity
    mean_connectivity = np.sum(W ** 2) - n ** 2 * connectivity_variance

    # 라플라시안 행렬 계산
    D = np.diag(degrees)
    L = D - W

    # 라플라시안 행렬의 고유값 계산
    eigenvalues = linalg.eigvalsh(L)
    eigenvalues.sort()  # 오름차순 정렬

    # 두 번째로 작은 고유값 (첫 번째는 항상 0 또는 0에 가까움)
    second_eigenvalue = eigenvalues[1] if len(eigenvalues) > 1 else 0

    return {
        'connectivity_variance': connectivity_variance,
        'mean_connectivity': mean_connectivity,
        'second_eigenvalue': second_eigenvalue
    }


def generate_random_graph(size, disconnected_probability=0.3):
    """
    랜덤 그래프를 생성합니다. disconnected_probability에 따라 연결이 끊어진 그래프도 생성합니다.

    Parameters:
    size (int): 그래프의 정점 수
    disconnected_probability (float): 그래프가 disconnected될 확률 (0~1)

    Returns:
    numpy.ndarray: 생성된 그래프의 weight matrix
    """
    matrix = np.zeros((size, size))

    # disconnected 그래프 생성 여부 결정
    is_disconnected = random.random() < disconnected_probability

    if is_disconnected:
        # 2~3개의 컴포넌트로 분할
        num_components = random.randint(2, 3)
        component_sizes = []

        # 각 컴포넌트의 크기 결정
        remaining_nodes = size
        for i in range(num_components - 1):
            # 최소 1개의 노드부터 남은 노드의 절반까지
            comp_size = random.randint(1, max(1, remaining_nodes // 2))
            component_sizes.append(comp_size)
            remaining_nodes -= comp_size

        component_sizes.append(remaining_nodes)  # 마지막 컴포넌트

        # 각 컴포넌트 내에서 랜덤 연결 생성
        start_idx = 0
        for comp_size in component_sizes:
            for i in range(start_idx, start_idx + comp_size):
                for j in range(start_idx, start_idx + comp_size):
                    if i != j:  # 자기 자신과의 연결 제외
                        matrix[i, j] = random.random()  # 0과 1 사이의 랜덤 가중치

            start_idx += comp_size
    else:
        # 일반 연결 그래프 생성
        for i in range(size):
            for j in range(size):
                if i != j:  # 자기 자신과의 연결 제외
                    # 밀도를 다양하게 하기 위해 일부 엣지만 생성
                    if random.random() < 0.7:
                        matrix[i, j] = random.random()

    return matrix


def generate_near_ucg(size, base_weight, noise_level=0.2):
    """
    UCG에 가까운 그래프를 생성합니다. 기본 가중치에 약간의 노이즈를 추가합니다.

    Parameters:
    size (int): 그래프의 정점 수
    base_weight (float): 기본 가중치 (0~1)
    noise_level (float): 노이즈 수준 (0~1)

    Returns:
    numpy.ndarray: UCG와 유사한 weight matrix
    """
    # 기본 UCG 생성
    matrix = np.ones((size, size)) * base_weight
    np.fill_diagonal(matrix, 0)  # 자기 자신과의 연결 제거

    # 노이즈 추가
    noise = np.random.uniform(-noise_level, noise_level, (size, size))
    noise = (noise + noise.T) / 2  # 대칭 노이즈 생성
    np.fill_diagonal(noise, 0)  # 대각선 요소는 노이즈 없음

    # 노이즈 적용
    matrix = matrix + noise * base_weight

    # 0-1 범위로 클리핑
    matrix = np.clip(matrix, 0, 1)

    return matrix


def generate_ucg(size, weight):
    """
    Uniformly Complete Graph (UCG)를 생성합니다.

    Parameters:
    size (int): 그래프의 정점 수
    weight (float): 모든 엣지의 가중치 (0~1)

    Returns:
    numpy.ndarray: UCG의 weight matrix
    """
    matrix = np.ones((size, size)) * weight
    np.fill_diagonal(matrix, 0)  # 자기 자신과의 연결 제거
    return matrix


def generate_special_disconnected_graphs(size, num_variants=10):
    """
    특수한 disconnected 그래프들을 생성합니다.
    1. 하나의 노드만 분리된 UCG
    2. 두 개의 UCG가 결합된 형태
    3. 세 개의 UCG가 결합된 형태

    Parameters:
    size (int): 그래프의 정점 수
    num_variants (int): 각 타입 별 생성할 그래프 수

    Returns:
    list: 생성된 그래프와 타입 정보를 담은 리스트
    """
    graphs = []

    # 최소 크기 체크
    if size < 4:
        print("Warning: 특수 disconnected 그래프를 생성하려면 그래프 크기가 최소 4 이상이어야 합니다.")
        return graphs

    # 1. 하나의 노드만 분리된 UCG
    print("하나의 노드만 분리된 UCG 생성...")
    for i in range(num_variants):
        # 기본 가중치 랜덤 설정
        weight = np.random.uniform(0.3, 0.9)

        # 기본 UCG 생성
        matrix = np.ones((size, size)) * weight
        np.fill_diagonal(matrix, 0)  # 자기 자신과의 연결 제거

        # 랜덤으로 하나의 노드 선택
        isolated_node = np.random.randint(0, size)

        # 해당 노드를 완전히 분리 (모든 연결 제거)
        matrix[isolated_node, :] = 0
        matrix[:, isolated_node] = 0

        graphs.append({
            'matrix': matrix,
            'type': 'isolated-node-ucg',
            'weight': weight,
            'isolated_node': isolated_node
        })

    # 2. 두 개의 UCG가 결합된 형태
    if size >= 4:  # 최소한 각 컴포넌트에 2개 이상의 노드가 필요
        print("두 개의 UCG가 결합된 형태 생성...")
        for i in range(num_variants):
            # 두 개의 컴포넌트 크기 결정
            # 최소한 각 컴포넌트에 노드가 2개 이상 있어야 함
            comp1_size = np.random.randint(2, size - 1)
            comp2_size = size - comp1_size

            # 두 컴포넌트의 가중치 설정
            weight1 = np.random.uniform(0.3, 0.9)
            weight2 = np.random.uniform(0.3, 0.9)

            # 기본 행렬 생성
            matrix = np.zeros((size, size))

            # 첫 번째 컴포넌트 설정
            for i in range(comp1_size):
                for j in range(comp1_size):
                    if i != j:
                        matrix[i, j] = weight1

            # 두 번째 컴포넌트 설정
            for i in range(comp1_size, size):
                for j in range(comp1_size, size):
                    if i != j:
                        matrix[i, j] = weight2

            graphs.append({
                'matrix': matrix,
                'type': 'two-ucg',
                'weight1': weight1,
                'weight2': weight2,
                'comp1_size': comp1_size,
                'comp2_size': comp2_size
            })

    # 3. 세 개의 UCG가 결합된 형태
    if size >= 6:  # 최소한 각 컴포넌트에 2개 이상의 노드가 필요
        print("세 개의 UCG가 결합된 형태 생성...")
        for i in range(num_variants):
            # 세 개의 컴포넌트 크기 결정
            # 랜덤하게 나누되, 최소 2개씩은 할당
            comp1_size = np.random.randint(2, size - 4)
            remaining = size - comp1_size
            comp2_size = np.random.randint(2, remaining - 1)
            comp3_size = remaining - comp2_size

            # 세 컴포넌트의 가중치 설정
            weight1 = np.random.uniform(0.3, 0.9)
            weight2 = np.random.uniform(0.3, 0.9)
            weight3 = np.random.uniform(0.3, 0.9)

            # 기본 행렬 생성
            matrix = np.zeros((size, size))

            # 첫 번째 컴포넌트 설정
            for i in range(comp1_size):
                for j in range(comp1_size):
                    if i != j:
                        matrix[i, j] = weight1

            # 두 번째 컴포넌트 설정
            for i in range(comp1_size, comp1_size + comp2_size):
                for j in range(comp1_size, comp1_size + comp2_size):
                    if i != j:
                        matrix[i, j] = weight2

            # 세 번째 컴포넌트 설정
            for i in range(comp1_size + comp2_size, size):
                for j in range(comp1_size + comp2_size, size):
                    if i != j:
                        matrix[i, j] = weight3

            graphs.append({
                'matrix': matrix,
                'type': 'three-ucg',
                'weight1': weight1,
                'weight2': weight2,
                'weight3': weight3,
                'comp1_size': comp1_size,
                'comp2_size': comp2_size,
                'comp3_size': comp3_size
            })

    # 4. UCG + 약한 연결 노드들
    print("UCG + 약한 연결 노드들 생성...")
    for i in range(num_variants):
        # 메인 UCG 크기와 가중치 결정
        main_ucg_size = np.random.randint(size // 2, size - 1)
        weak_nodes = size - main_ucg_size

        weight = np.random.uniform(0.5, 0.9)
        weak_weight = np.random.uniform(0.05, 0.2)

        # 기본 행렬 생성
        matrix = np.zeros((size, size))

        # 메인 UCG 설정
        for i in range(main_ucg_size):
            for j in range(main_ucg_size):
                if i != j:
                    matrix[i, j] = weight

        # 약한 연결 노드들 설정 - 메인 UCG 노드들과만 약하게 연결
        for i in range(main_ucg_size, size):
            for j in range(main_ucg_size):
                matrix[i, j] = weak_weight
                matrix[j, i] = weak_weight

        graphs.append({
            'matrix': matrix,
            'type': 'ucg-with-weak-nodes',
            'main_weight': weight,
            'weak_weight': weak_weight,
            'main_ucg_size': main_ucg_size,
            'weak_nodes': weak_nodes
        })

    return graphs
    # disconnected 그래프 생성 여부 결정
    is_disconnected = random.random() < disconnected_probability

    if is_disconnected:
        # 2~3개의 컴포넌트로 분할
        num_components = random.randint(2, 3)
        component_sizes = []

        # 각 컴포넌트의 크기 결정
        remaining_nodes = size
        for i in range(num_components - 1):
            # 최소 1개의 노드부터 남은 노드의 절반까지
            comp_size = random.randint(1, max(1, remaining_nodes // 2))
            component_sizes.append(comp_size)
            remaining_nodes -= comp_size

        component_sizes.append(remaining_nodes)  # 마지막 컴포넌트

        # 각 컴포넌트 내에서 랜덤 연결 생성
        start_idx = 0
        for comp_size in component_sizes:
            for i in range(start_idx, start_idx + comp_size):
                for j in range(start_idx, start_idx + comp_size):
                    if i != j:  # 자기 자신과의 연결 제외
                        matrix[i, j] = random.random()  # 0과 1 사이의 랜덤 가중치

            start_idx += comp_size
    else:
        # 일반 연결 그래프 생성
        for i in range(size):
            for j in range(size):
                if i != j:  # 자기 자신과의 연결 제외
                    # 밀도를 다양하게 하기 위해 일부 엣지만 생성
                    if random.random() < 0.7:
                        matrix[i, j] = random.random()

    return matrix


def generate_near_ucg(size, base_weight, noise_level=0.2):
    """
    UCG에 가까운 그래프를 생성합니다. 기본 가중치에 약간의 노이즈를 추가합니다.

    Parameters:
    size (int): 그래프의 정점 수
    base_weight (float): 기본 가중치 (0~1)
    noise_level (float): 노이즈 수준 (0~1)

    Returns:
    numpy.ndarray: UCG와 유사한 weight matrix
    """
    # 기본 UCG 생성
    matrix = np.ones((size, size)) * base_weight
    np.fill_diagonal(matrix, 0)  # 자기 자신과의 연결 제거

    # 노이즈 추가
    noise = np.random.uniform(-noise_level, noise_level, (size, size))
    noise = (noise + noise.T) / 2  # 대칭 노이즈 생성
    np.fill_diagonal(noise, 0)  # 대각선 요소는 노이즈 없음

    # 노이즈 적용
    matrix = matrix + noise * base_weight

    # 0-1 범위로 클리핑
    matrix = np.clip(matrix, 0, 1)

    return matrix


def generate_ucg(size, weight):
    """
    Uniformly Complete Graph (UCG)를 생성합니다.

    Parameters:
    size (int): 그래프의 정점 수
    weight (float): 모든 엣지의 가중치 (0~1)

    Returns:
    numpy.ndarray: UCG의 weight matrix
    """
    matrix = np.ones((size, size)) * weight
    np.fill_diagonal(matrix, 0)  # 자기 자신과의 연결 제거
    return matrix


def visualize_graph_metrics(size, num_graphs=100, disconnected_prob=0.3, show_ucg=True, ucg_variants=30,
                            special_disconnected=10):
    """
    다양한 그래프에 대한 메트릭을 계산하고 시각화합니다.

    Parameters:
    size (int): 그래프의 정점 수
    num_graphs (int): 생성할 랜덤 그래프 수
    disconnected_prob (float): disconnected 그래프 비율
    show_ucg (bool): UCG 포함 여부
    ucg_variants (int): UCG에 가까운 그래프 변형의 수
    special_disconnected (int): 특수 disconnected 그래프 수

    Returns:
    list: 계산된 메트릭 데이터
    """
    # 데이터 저장용 리스트
    data = []

    # 랜덤 그래프 생성 및 메트릭 계산
    print(f"Generating {num_graphs} random graphs...")
    for i in range(num_graphs):
        matrix = generate_random_graph(size, disconnected_prob)
        try:
            metrics = compute_graph_metrics(matrix)
            data.append({
                'id': i,
                'mean_connectivity': metrics['mean_connectivity'],
                'connectivity_variance': metrics['connectivity_variance'],
                'second_eigenvalue': metrics['second_eigenvalue'],
                'type': 'random'
            })
        except Exception as e:
            print(f"Error computing metrics for graph {i}: {e}")

    # UCG 생성 및 메트릭 계산
    if show_ucg:
        print("Generating UCGs with different weights...")
        for w in np.arange(0.1, 1.1, 0.1):
            matrix = generate_ucg(size, w)
            try:
                metrics = compute_graph_metrics(matrix)
                data.append({
                    'id': f'ucg-{w:.1f}',
                    'mean_connectivity': metrics['mean_connectivity'],
                    'connectivity_variance': metrics['connectivity_variance'],
                    'second_eigenvalue': metrics['second_eigenvalue'],
                    'type': 'ucg',
                    'weight': w
                })
            except Exception as e:
                print(f"Error computing metrics for UCG with weight {w}: {e}")

        # UCG에 가까운 그래프 생성
        if ucg_variants > 0:
            print(f"Generating {ucg_variants} variants of UCG-like graphs...")
            for i in range(ucg_variants):
                # 랜덤 기본 가중치
                w = np.random.uniform(0.2, 0.9)

                noise = np.random.uniform(0.05, 0.4)
                matrix = generate_near_ucg(size, w, noise)

                try:
                    metrics = compute_graph_metrics(matrix)
                    data.append({
                        'id': f'near-ucg-{i}',
                        'mean_connectivity': metrics['mean_connectivity'],
                        'connectivity_variance': metrics['connectivity_variance'],
                        'second_eigenvalue': metrics['second_eigenvalue'],
                        'type': 'near-ucg',
                        'base_weight': w,
                        'noise': noise
                    })
                except Exception as e:
                    print(f"Error computing metrics for near-UCG graph {i}: {e}")

    # 특수 disconnected 그래프 생성
    if special_disconnected > 0:
        print(f"Generating special disconnected graphs...")
        special_graphs = generate_special_disconnected_graphs(size, special_disconnected)

        for graph_info in special_graphs:
            matrix = graph_info['matrix']
            graph_type = graph_info['type']

            try:
                metrics = compute_graph_metrics(matrix)
                data.append({
                    'id': f'{graph_type}-{len(data)}',
                    'mean_connectivity': metrics['mean_connectivity'],
                    'connectivity_variance': metrics['connectivity_variance'],
                    'second_eigenvalue': metrics['second_eigenvalue'],
                    'type': graph_type,
                    **{k: v for k, v in graph_info.items() if k != 'matrix' and k != 'type'}
                })
            except Exception as e:
                print(f"Error computing metrics for special graph {graph_type}: {e}")

    # 데이터 준비
    x_values = [item['mean_connectivity'] for item in data]
    y_values = [item['connectivity_variance'] for item in data]

    # 색상 매핑을 위한 값 추출
    color_values = [item['second_eigenvalue'] for item in data]

    # 색상 범위 정규화 - second eigenvalue는 항상 0 이상
    max_eigenvalue = max(color_values)

    # 모든 eigenvalue를 0과 1 사이로 정규화
    if max_eigenvalue > 0:
        color_values = [val for val in color_values]
    else:
        # 모든 그래프가 disconnected인 경우
        color_values = [0 for _ in color_values]

    # 정규화된 값 설정
    norm = Normalize(vmin=0, vmax=1)

    # 그래프 타입에 따른 마커 설정
    markers = {
        'random': 'o',
        'ucg': 'o',
        'near-ucg': 'o',
        'isolated-node-ucg': 'o',  # 다이아몬드
        'two-ucg': 'o',  # 오각형
        'three-ucg': 'o',  # 육각형
        'ucg-with-weak-nodes': 'o'  # 플러스
    }

    # 시각화
    plt.figure(figsize=(14, 10))

    # 각 타입에 대해 별도로 플롯
    for type_name in markers.keys():
        # 해당 타입의 데이터 인덱스 찾기
        indices = [i for i, item in enumerate(data) if item['type'] == type_name]

        if indices:
            # 해당 타입의 데이터 추출
            type_x = [x_values[i] for i in indices]
            type_y = [y_values[i] for i in indices]
            type_colors = [color_values[i] for i in indices]

            # 타입별 플롯
            scatter = plt.scatter(
                type_x,
                type_y,
                c=type_colors,  # 색상값으로 사용
                cmap='cool',  # 색상 맵
                norm=norm,  # 색상 정규화
                marker=markers[type_name],
                s=1 if type_name != 'random' else 1,  # 랜덤이 아닌 그래프는 더 크게
                alpha=0.8,
                edgecolors='black' if type_name == 'ucg' else None  # UCG는 테두리 추가
            )

    # 컬러바 추가
    cbar = plt.colorbar()
    cbar.set_label('Normalized Second Eigenvalue (Algebraic Connectivity)')

    # 축 레이블 및 제목 설정
    plt.xlabel('Mean Connectivity')
    plt.ylabel('Connectivity Variance')
    plt.title(f'Graph Metrics Visualization (Size: {size}, Graphs: {len(data)})')

    # 범례 추가
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=1, label='Random Graph'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=1, label='UCG'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=1, label='Near-UCG'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=1, label='UCG with Isolated Node'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=1, label='Two Connected UCGs'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=1, label='Three Connected UCGs'),
        Line2D([0], [0], marker='o', color='gray', markersize=1, label='UCG with Weak Nodes')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    # 그리드 추가
    plt.grid(True, linestyle='--', alpha=0.7)

    # 그래프 저장 및 표시
    plt.tight_layout()
    plt.savefig('graph_metrics_visualization.png', dpi=300)
    plt.show()

    return data


def analyze_results(data):
    """
    계산된 메트릭 결과를 분석합니다.

    Parameters:
    data (list): 계산된 메트릭 데이터
    """
    # 타입별로 데이터 분리
    random_data = [item for item in data if item['type'] == 'random']
    ucg_data = [item for item in data if item['type'] == 'ucg']
    near_ucg_data = [item for item in data if item['type'] == 'near-ucg']

    # 통계 계산
    if random_data:
        random_mean_conn = np.mean([item['mean_connectivity'] for item in random_data])
        random_conn_var = np.mean([item['connectivity_variance'] for item in random_data])
        random_second_eig = np.mean([item['second_eigenvalue'] for item in random_data])

        random_max_eig = max([item['second_eigenvalue'] for item in random_data])
        random_max_eig_idx = [item['second_eigenvalue'] for item in random_data].index(random_max_eig)
        random_max_item = random_data[random_max_eig_idx]

        print("\n랜덤 그래프 통계:")
        print(f"평균 Mean Connectivity: {random_mean_conn:.4f}")
        print(f"평균 Connectivity Variance: {random_conn_var:.4f}")
        print(f"평균 Second Eigenvalue: {random_second_eig:.4f}")
        print(f"최대 Second Eigenvalue: {random_max_eig:.4f}")
        print(f"최대 Second Eigenvalue를 가진 그래프 정보:")
        print(f"  - Mean Connectivity: {random_max_item['mean_connectivity']:.4f}")
        print(f"  - Connectivity Variance: {random_max_item['connectivity_variance']:.4f}")

    if ucg_data:
        ucg_mean_conn = np.mean([item['mean_connectivity'] for item in ucg_data])
        ucg_conn_var = np.mean([item['connectivity_variance'] for item in ucg_data])
        ucg_second_eig = np.mean([item['second_eigenvalue'] for item in ucg_data])

        ucg_max_eig = max([item['second_eigenvalue'] for item in ucg_data])
        ucg_max_eig_idx = [item['second_eigenvalue'] for item in ucg_data].index(ucg_max_eig)
        ucg_max_item = ucg_data[ucg_max_eig_idx]

        print("\nUCG 통계:")
        print(f"평균 Mean Connectivity: {ucg_mean_conn:.4f}")
        print(f"평균 Connectivity Variance: {ucg_conn_var:.4f}")
        print(f"평균 Second Eigenvalue: {ucg_second_eig:.4f}")
        print(f"최대 Second Eigenvalue: {ucg_max_eig:.4f}")
        print(f"최대 Second Eigenvalue를 가진 UCG 정보:")
        print(f"  - 가중치: {ucg_max_item.get('weight', 'N/A')}")
        print(f"  - Mean Connectivity: {ucg_max_item['mean_connectivity']:.4f}")
        print(f"  - Connectivity Variance: {ucg_max_item['connectivity_variance']:.4f}")

    if near_ucg_data:
        near_ucg_mean_conn = np.mean([item['mean_connectivity'] for item in near_ucg_data])
        near_ucg_conn_var = np.mean([item['connectivity_variance'] for item in near_ucg_data])
        near_ucg_second_eig = np.mean([item['second_eigenvalue'] for item in near_ucg_data])

        near_ucg_max_eig = max([item['second_eigenvalue'] for item in near_ucg_data])
        near_ucg_max_eig_idx = [item['second_eigenvalue'] for item in near_ucg_data].index(near_ucg_max_eig)
        near_ucg_max_item = near_ucg_data[near_ucg_max_eig_idx]

        print("\nNear-UCG 통계:")
        print(f"평균 Mean Connectivity: {near_ucg_mean_conn:.4f}")
        print(f"평균 Connectivity Variance: {near_ucg_conn_var:.4f}")
        print(f"평균 Second Eigenvalue: {near_ucg_second_eig:.4f}")
        print(f"최대 Second Eigenvalue: {near_ucg_max_eig:.4f}")
        print(f"최대 Second Eigenvalue를 가진 Near-UCG 정보:")
        print(f"  - 기본 가중치: {near_ucg_max_item.get('base_weight', 'N/A'):.2f}")
        print(f"  - 노이즈 수준: {near_ucg_max_item.get('noise', 'N/A'):.2f}")
        print(f"  - Mean Connectivity: {near_ucg_max_item['mean_connectivity']:.4f}")
        print(f"  - Connectivity Variance: {near_ucg_max_item['connectivity_variance']:.4f}")

    # 전체 데이터 관계 분석
    x = np.array([item['mean_connectivity'] for item in data])
    y = np.array([item['connectivity_variance'] for item in data])
    z = np.array([item['second_eigenvalue'] for item in data])

    # 상관 계수 계산
    corr_mean_var = np.corrcoef(x, y)[0, 1]
    corr_mean_eig = np.corrcoef(x, z)[0, 1]
    corr_var_eig = np.corrcoef(y, z)[0, 1]

    print("\n상관 관계 분석:")
    print(f"Mean Connectivity와 Connectivity Variance 간 상관계수: {corr_mean_var:.4f}")
    print(f"Mean Connectivity와 Second Eigenvalue 간 상관계수: {corr_mean_eig:.4f}")
    print(f"Connectivity Variance와 Second Eigenvalue 간 상관계수: {corr_var_eig:.4f}")
    corr_mean_var = np.corrcoef(x, y)[0, 1]
    corr_mean_eig = np.corrcoef(x, z)[0, 1]
    corr_var_eig = np.corrcoef(y, z)[0, 1]

    print("\n상관 관계 분석:")
    print(f"Mean Connectivity와 Connectivity Variance 간 상관계수: {corr_mean_var:.4f}")
    print(f"Mean Connectivity와 Second Eigenvalue 간 상관계수: {corr_mean_eig:.4f}")
    print(f"Connectivity Variance와 Second Eigenvalue 간 상관계수: {corr_var_eig:.4f}")


if __name__ == "__main__":
    # 사용자 입력 받기
    try:
        size = int(input("그래프 크기(정점 수)를 입력하세요 (기본값: 5): ") or 5)
        num_graphs = int(input("생성할 랜덤 그래프 수를 입력하세요 (기본값: 100): ") or 100)
        disconnected_prob = float(input("Disconnected 그래프 생성 확률을 입력하세요 (0~1, 기본값: 0.3): ") or 0.3)

        show_ucg_input = input("UCG를 포함하시겠습니까? (y/n, 기본값: y): ").lower() or 'y'
        show_ucg = show_ucg_input == 'y'

        if show_ucg:
            ucg_variants = int(input("UCG에 가까운 그래프 변형 수를 입력하세요 (기본값: 30): ") or 30)
        else:
            ucg_variants = 0

        special_disconnected = int(input("특수 disconnected 그래프 수를 입력하세요 (타입당, 기본값: 10): ") or 10)

        # 시각화 실행
        data = visualize_graph_metrics(size, num_graphs, disconnected_prob, show_ucg, ucg_variants,
                                       special_disconnected)

        print(f"\n총 {len(data)}개의 그래프에 대한 메트릭을 계산했습니다.")

        # 결과 분석
        analyze_results(data)

    except ValueError as e:
        print(f"입력 오류: {e}")
        print("기본값으로 실행합니다.")
        try:
            data = visualize_graph_metrics(5, 100, 0.3, True, 30, 10)
            analyze_results(data)
        except Exception as e:
            print(f"기본 실행 중 오류 발생: {e}")