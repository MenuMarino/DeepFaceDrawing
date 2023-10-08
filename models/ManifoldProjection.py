import pickle

import numpy as np
import torch


class Module:
    def __init__(self, tree_path='./tree.pickle'):
        with open(tree_path, 'rb') as f:
            self.tree = pickle.load(f)

    def project(self, point, k):
        knn_val, knn_ind = self.tree.query(point, k=k)
        knn_val = knn_val.flatten()
        knn_ind = knn_ind.flatten()
        
        neighbors = [self.tree.data[index] for index in knn_ind]
        
        weights = 1 / (knn_val**2 + 1e-10)  # Usamos el cuadrado de la distancia
        weights /= weights.sum()

        projected_point = np.zeros_like(neighbors[0])
        for i, neighbor in enumerate(neighbors):
            w = weights[i]
            projected_point += w * neighbor
        return projected_point
    
    def project_dict(self, input_dict, device='cuda', k=10):
      projected_dict = {}
      for key, value in input_dict.items():
          value_numpy = value.cpu().numpy()
          projected_value = self.project(value_numpy, k)
          projected_tensor = torch.Tensor(projected_value).unsqueeze(0).to(device)
          projected_dict[key] = projected_tensor
      return projected_dict

