import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import random


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


def visualize_graph_metrics(size, num_graphs=100, disconnected_prob=0.3, show_ucg=True):
    """
    다양한 그래프에 대한 메트릭을 계산하고 시각화합니다.

    Parameters:
    size (int): 그래프의 정점 수
    num_graphs (int): 생성할 랜덤 그래프 수
    disconnected_prob (float): disconnected 그래프 비율
    show_ucg (bool): UCG 포함 여부
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

    # 데이터 추출
    mean_connectivity = [item['mean_connectivity'] for item in data]
    connectivity_variance = [item['connectivity_variance'] for item in data]
    second_eigenvalue = [item['second_eigenvalue'] for item in data]
    types = [item['type'] for item in data]

    # 색상 정규화 (second eigenvalue 기준)
    max_eigenvalue = max(second_eigenvalue)
    colors = []
    for eigenvalue in second_eigenvalue:
        # 정규화된 값 (0-1)
        normalized = eigenvalue / max_eigenvalue if max_eigenvalue > 0 else 0
        # 파란색(낮은 값)에서 빨간색(높은 값)으로 그라데이션
        colors.append((normalized, 0, 1 - normalized))

    # 그래프 타입 별 마커 설정
    markers = []
    for t in types:
        if t == 'ucg':
            markers.append('s')  # UCG는 사각형
        else:
            markers.append('o')  # 랜덤 그래프는 원

    # 시각화
    plt.figure(figsize=(12, 8))

    # 각 데이터 포인트를 개별적으로 플롯
    for i in range(len(data)):
        plt.scatter(
            mean_connectivity[i],
            connectivity_variance[i],
            color=colors[i],
            marker=markers[i],
            s=100 if types[i] == 'ucg' else 50,  # UCG는 더 크게
            alpha=0.8,
            edgecolors='black' if types[i] == 'ucg' else None  # UCG는 테두리 추가
        )

    # 컬러바 추가
    sm = plt.cm.ScalarMappable(cmap=plt.cm.cool)
    sm.set_array(second_eigenvalue)
    cbar = plt.colorbar(sm)
    cbar.set_label('Second Eigenvalue')

    # 축 레이블 및 제목 설정
    plt.xlabel('Mean Connectivity')
    plt.ylabel('Connectivity Variance')
    plt.title(f'Graph Metrics Visualization (Size: {size}, Graphs: {num_graphs})')

    # 범례 추가
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Random Graph'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='UCG')
    ]
    plt.legend(handles=legend_elements)

    # 그리드 추가
    plt.grid(True, linestyle='--', alpha=0.7)

    # 그래프 저장 및 표시
    plt.tight_layout()
    plt.savefig('graph_metrics_visualization.png', dpi=300)
    plt.show()

    return data


if __name__ == "__main__":
    # 사용자 입력 받기
    try:
        size = int(input("그래프 크기(정점 수)를 입력하세요 (기본값: 5): ") or 5)
        num_graphs = int(input("생성할 랜덤 그래프 수를 입력하세요 (기본값: 100): ") or 100)
        disconnected_prob = float(input("Disconnected 그래프 생성 확률을 입력하세요 (0~1, 기본값: 0.3): ") or 0.3)

        show_ucg_input = input("UCG를 포함하시겠습니까? (y/n, 기본값: y): ").lower() or 'y'
        show_ucg = show_ucg_input == 'y'

        # 시각화 실행
        data = visualize_graph_metrics(size, num_graphs, disconnected_prob, show_ucg)

        print(f"\n총 {len(data)}개의 그래프에 대한 메트릭을 계산했습니다.")

    except ValueError as e:
        print(f"입력 오류: {e}")
        print("기본값으로 실행합니다.")
        visualize_graph_metrics(5, 100, 0.3, True)