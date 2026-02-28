import torch


class PrinSubspace:
    def __init__(self, prin_subspace_name_list: list, log_txt):
        self.prin_subspace_dict = {name: [] for name in prin_subspace_name_list}
        self.log_txt = log_txt

    def update_prin_subspace(self, mat_list_dict: dict, threshold_dict: dict | float):
        if isinstance(threshold_dict, float):
            threshold = threshold_dict
            threshold_dict = {name: threshold for name in self.prin_subspace_dict.keys()}
        if set(mat_list_dict.keys()) != set(self.prin_subspace_dict.keys()) or set(threshold_dict.keys()) != set(self.prin_subspace_dict.keys()):
            raise ValueError("Keys in mat_list_dict or threshold_dict do not match prin_subspace_dict")
        for k in self.prin_subspace_dict.keys():
            if threshold_dict[k] > 1e-8:
                self._update_prin_subspace(mat_list_dict[k], self.prin_subspace_dict[k], threshold_dict[k])

    def _update_prin_subspace(self, mat_list, prin_subspace_layers, threshold):
        if len(prin_subspace_layers) == 0:
            # After First Task 
            for i in range(len(mat_list)):
                activation = mat_list[i].clone().detach().float()
                U, S, Vh = torch.linalg.svd(activation, full_matrices=False)
                
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = torch.sum(torch.cumsum(sval_ratio, dim=0) < threshold).item()
                prin_subspace_layers.append(U[:,0:max(r,1)])
        else:
            for i in range(len(mat_list)):
                activation = mat_list[i].clone().detach().float()
                U1, S1, Vh1 = torch.linalg.svd(activation, full_matrices=False)
                sval_total = (S1**2).sum()
                
                feature_tensor = prin_subspace_layers[i].clone().detach().float()
                act_hat = activation - torch.matmul(torch.matmul(feature_tensor, feature_tensor.t()), activation)
                U, S, Vh = torch.linalg.svd(act_hat, full_matrices=False)
                
                sval_hat = (S**2).sum()
                sval_ratio = (S**2)/sval_total               
                accumulated_sval = (sval_total-sval_hat)/sval_total
                
                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated_sval < threshold:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    print('Skip Updating prin_subspace for layer: {}'.format(i+1)) 
                    continue
                # update prin_subspace
                Ui = torch.hstack((feature_tensor, U[:,0:r]))
                if Ui.shape[1] > Ui.shape[0]:
                    prin_subspace_layers[i] = Ui[:,0:Ui.shape[0]]
                else:
                    prin_subspace_layers[i] = Ui
        
        for i in range(len(prin_subspace_layers)):
            log = 'Layer {} : {}/{}'.format(i+1, prin_subspace_layers[i].shape[1], prin_subspace_layers[i].shape[0])
            self.log_txt(log)