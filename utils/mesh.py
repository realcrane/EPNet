import numpy as np
import tensorflow as tf
from scipy.sparse import coo_matrix


def triangulate(faces):
    triangles = np.int32(
        [triangle for polygon in faces for triangle in _triangulate_recursive(polygon)]
    )
    return triangles


def _triangulate_recursive(face):
    if len(face) == 3:
        return [face]
    else:
        return [face[:3]] + _triangulate_recursive([face[0], *face[2:]])


def faces_to_edges_and_adjacency(faces):
    edges = dict()
    for fidx, face in enumerate(faces):
        for i, v in enumerate(face):
            nv = face[(i + 1) % len(face)]
            edge = tuple(sorted([v, nv]))
            if edge not in edges:
                edges[edge] = []
            edges[edge] += [fidx]
    face_adjacency = []
    face_adjacency_edges = []
    edge_adjacency_index = []
    vertices_neighbours = [[] for _ in range(np.max(faces) + 1)]
    vertices_neighbours_edges = [[] for _ in range(np.max(faces) + 1)]
    edges_neighbours = []
    max_vertices_neighbours_count = 0
    max_edges_neighbours_count = 0
    for index, (edge, face_list) in enumerate(edges.items()):
        for i in range(len(face_list) - 1):
            for j in range(i + 1, len(face_list)):
                face_adjacency += [[face_list[i], face_list[j]]]
                face_adjacency_edges += [edge]
                edge_adjacency_index += [index]
        vertices_neighbours[edge[0]].append(edge[1])
        vertices_neighbours_edges[edge[0]].append(index)
        if len(vertices_neighbours[edge[0]]) > max_vertices_neighbours_count:
            max_vertices_neighbours_count = len(vertices_neighbours[edge[0]])
        vertices_neighbours[edge[1]].append(edge[0])
        vertices_neighbours_edges[edge[1]].append(index)
        if len(vertices_neighbours[edge[1]]) > max_vertices_neighbours_count:
            max_vertices_neighbours_count = len(vertices_neighbours[edge[1]])

    for index, (edge, _) in enumerate(edges.items()):
        result = np.unique(np.concatenate([vertices_neighbours_edges[edge[0]], vertices_neighbours_edges[edge[1]]]))
        result = result[~np.isin(result, index)]
        if len(result) > max_edges_neighbours_count:
            max_edges_neighbours_count = len(result)
        edges_neighbours.append(result)

    edges = np.array([list(edge) for edge in edges.keys()], np.int32)
    vertices_neighbours = [np.array(vertex, np.int32) for vertex in vertices_neighbours]
    vertices_neighbours_edges = [np.array(edge, np.int32) for edge in vertices_neighbours_edges]
    face_adjacency = np.array(face_adjacency, np.int32)
    face_adjacency_edges = np.array(face_adjacency_edges, np.int32)
    edge_adjacency_index = np.array(edge_adjacency_index, np.int32)

    return edges, face_adjacency, face_adjacency_edges, edge_adjacency_index, \
        max_vertices_neighbours_count, vertices_neighbours, vertices_neighbours_edges, \
        max_edges_neighbours_count, edges_neighbours


def laplacian_matrix(faces):
    G = {}
    for face in faces:
        for i, v in enumerate(face):
            nv = face[(i + 1) % len(face)]
            if v not in G:
                G[v] = {}
            if nv not in G:
                G[nv] = {}
            G[v][nv] = 1
            G[nv][v] = 1
    return graph_laplacian(G)


def graph_laplacian(graph):
    row, col, data = [], [], []
    for v in graph:
        n = len(graph[v])
        row += [v] * n
        col += [u for u in graph[v]]
        data += [1.0 / n] * n
    return coo_matrix((data, (row, col)), shape=[len(graph)] * 2)


@tf.function
def face_normals(vertices, faces, normalized=True):
    input_shape = tf.shape(vertices)
    vertex_count = input_shape[-2]
    vertices_reshaped = tf.reshape(vertices, [-1, vertex_count, 3])

    v01 = tf.gather(vertices_reshaped, faces[:, 1], axis=1) - tf.gather(
        vertices_reshaped, faces[:, 0], axis=1)
    v12 = tf.gather(vertices_reshaped, faces[:, 2], axis=1) - tf.gather(
        vertices_reshaped, faces[:, 1], axis=1)

    normals = tf.linalg.cross(v01, v12)

    if normalized:
        normals = tf.math.l2_normalize(normals, axis=-1)

    face_count = tf.shape(normals)[1]
    output_shape = tf.concat([input_shape[:-2], [face_count, 3]], axis=0)
    return tf.reshape(normals, output_shape)


@tf.function
def vertex_normals(vertices, faces):
    input_shape = vertices.get_shape()
    batch_size = tf.reduce_prod(input_shape[:-2] or [1])
    vertices = tf.reshape(vertices, (-1, *input_shape[-2:]))
    mesh_normals = face_normals(vertices, faces, normalized=False)
    faces_batched = tf.stack(
        (
            tf.tile(tf.range(batch_size)[:, None, None], [1, *np.shape(faces)]),
            tf.tile(faces[None], [batch_size, 1, 1]),
        ),
        axis=-1,
    )
    mesh_normals = tf.tile(mesh_normals[:, :, None], [1, 1, 3, 1])
    vertex_normals = tf.zeros((batch_size, *input_shape[-2:]), tf.float32)
    vertex_normals = tf.tensor_scatter_nd_add(
        vertex_normals, faces_batched, mesh_normals
    )
    vertex_normals /= (
        tf.norm(vertex_normals, axis=-1, keepdims=True) + tf.keras.backend.epsilon()
    )
    vertex_normals = tf.reshape(vertex_normals, input_shape)
    return vertex_normals


def edge_lengths(vertices, edges):
    return np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=-1)


def dihedral_angle_adjacent_faces(normals, adjacency):
    normals0 = normals[adjacency[:, 0]]
    normals1 = normals[adjacency[:, 1]]
    cos = np.einsum("ab,ab->a", normals0, normals1)
    sin = np.linalg.norm(np.cross(normals0, normals1), axis=-1)
    return np.arctan2(sin, cos)


def dihedral_dir_angle_adjacent_faces(normals, adjacency, edge_adjacency_dir):
    normals0 = normals[adjacency[:, 0]]
    normals1 = normals[adjacency[:, 1]]

    cos = np.einsum("ab,ab->a", normals0, normals1)
    sin = np.linalg.norm(np.cross(normals0, normals1), axis=-1)
    angle = np.arctan2(sin, cos)

    orientation = np.sum(edge_adjacency_dir * np.cross(normals0, normals1), axis=1)
    return np.where(orientation < 0, -angle, angle)


def edge_dir_adjacent(edges, adj_edge_id, vertices):
    adj_edge = edges[adj_edge_id]
    adj_vertex_0 = vertices[adj_edge[:, 0]]
    adj_vertex_1 = vertices[adj_edge[:, 1]]
    adj_edge_dir = adj_vertex_1 - adj_vertex_0

    norms = np.linalg.norm(adj_edge_dir, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return adj_edge_dir / norms


def vertex_area(vertices, faces):
    v01 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    v12 = vertices[faces[:, 2]] - vertices[faces[:, 1]]
    face_areas = np.linalg.norm(np.cross(v01, v12), axis=-1)
    vertex_areas = np.zeros((vertices.shape[0],), np.float32)
    for i, face in enumerate(faces):
        vertex_areas[face] += face_areas[i]
    vertex_areas *= 1 / 6
    total_area = vertex_areas.sum()
    return vertex_areas, face_areas, total_area


def lbs(vertices, matrices, blend_weights=None):
    matrices = tf.cast(matrices, tf.float32)
    vertices = tf.cast(vertices, tf.float32)

    matrix_shape = tf.shape(matrices)
    matrices = tf.reshape(matrices, tf.concat([matrix_shape[:-2], [12]], axis=0))

    if blend_weights is not None:
        blend_weights = tf.cast(blend_weights, tf.float32)
        matrices = blend_weights @ matrices

    matrix_shape = tf.shape(matrices)
    matrices = tf.reshape(matrices, tf.concat([matrix_shape[:-1], [3, 4]], axis=0))

    rotations, translations = tf.split(matrices, [3, 1], axis=-1)

    vertices = tf.matmul(rotations, vertices[..., None])
    vertices += translations

    return vertices[..., 0]

